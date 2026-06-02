#!/bin/bash
# =============================================================================
# edu_rl.sh —— 教育「题目级」推荐全链路一键脚本（造数 demo 版）
#
# 串联：① 造模拟数据 → ② 转换为训练 CSV → ③ SFT → ④ RL(ZPD reward) → ⑤ 评估
#
# 用法：
#   bash edu_rl.sh                 # 跑全部阶段
#   STAGE_SFT=0 STAGE_RL=0 STAGE_EVAL=0 bash edu_rl.sh   # 只造数+转换
#   STAGE_DATA=0 STAGE_CONVERT=0 bash edu_rl.sh          # 跳过造数，直接训练
# =============================================================================
set -e

# ----------------------------- 可配置区 --------------------------------------
MODEL_PATH=${MODEL_PATH:-"your_model_path"}   # 基座模型路径（必改）
NUM_GPUS=${NUM_GPUS:-1}                        # 训练用 GPU 数
MAIN_PORT=${MAIN_PORT:-29503}

DATASET_NAME="Edu_Questions"
CATEGORY="Edu_Questions"
SEED=42

# 造数规模
NUM_KP=${NUM_KP:-30}            # 知识点数（= 第一层 SID 取值范围）
Q_PER_KP=${Q_PER_KP:-8}        # 每个知识点的题目数
NUM_STUDENTS=${NUM_STUDENTS:-800}

# 目录
RAW_DIR="data/Edu_raw"                          # 造数原始输出目录
RAW_DATASET_DIR="${RAW_DIR}/${DATASET_NAME}"     # item/index/inter 所在目录
PROC_DIR="data/Edu"                             # 转换后 train/valid/test/info 目录

SID_INDEX_PATH="${RAW_DATASET_DIR}/${DATASET_NAME}.index.json"
ITEM_META_PATH="${RAW_DATASET_DIR}/${DATASET_NAME}.item.json"

SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-"output_dir/edu_sft"}
RL_OUTPUT_DIR=${RL_OUTPUT_DIR:-"output_dir/edu_rl"}

# RL reward 类型：zpd（纯 ZPD）或 ranking_zpd（命中目标 + ZPD 联合）
REWARD_TYPE=${REWARD_TYPE:-"ranking_zpd"}

# 阶段开关（1=执行，0=跳过）
STAGE_DATA=${STAGE_DATA:-1}
STAGE_CONVERT=${STAGE_CONVERT:-1}
STAGE_SFT=${STAGE_SFT:-1}
STAGE_RL=${STAGE_RL:-1}
STAGE_EVAL=${STAGE_EVAL:-1}

export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled

echo "=============================================================="
echo " 教育推荐全链路  | model=${MODEL_PATH}  gpus=${NUM_GPUS}"
echo " reward_type=${REWARD_TYPE}"
echo "=============================================================="

# --------------------------- ① 造模拟数据 ------------------------------------
if [[ "$STAGE_DATA" == "1" ]]; then
    echo ">>> [1/5] 造模拟数据 ..."
    python data/edu_data_process.py \
        --out_dir ${RAW_DIR} \
        --dataset_name ${DATASET_NAME} \
        --num_kp ${NUM_KP} \
        --q_per_kp ${Q_PER_KP} \
        --num_students ${NUM_STUDENTS} \
        --seed ${SEED}
    echo ">>> 造数完成。"
fi

# --------------------------- ② 转换为训练 CSV --------------------------------
if [[ "$STAGE_CONVERT" == "1" ]]; then
    echo ">>> [2/5] 转换数据集 -> CSV/info ..."
    python convert_dataset.py \
        --dataset_name ${DATASET_NAME} \
        --data_dir ${RAW_DATASET_DIR} \
        --output_dir ${PROC_DIR} \
        --category ${CATEGORY} \
        --seed ${SEED}
    echo ">>> 转换完成。"
fi

# 训练/评估所需文件（转换产物）
train_file=$(ls ${PROC_DIR}/train/${DATASET_NAME}*.csv 2>/dev/null | head -1)
valid_file=$(ls ${PROC_DIR}/valid/${DATASET_NAME}*.csv 2>/dev/null | head -1)
test_file=$(ls ${PROC_DIR}/test/${DATASET_NAME}*.csv 2>/dev/null | head -1)
info_file=$(ls ${PROC_DIR}/info/${DATASET_NAME}*.txt 2>/dev/null | head -1)
echo "train=${train_file}"
echo "valid=${valid_file}"
echo "test =${test_file}"
echo "info =${info_file}"

# --------------------------- ③ SFT 冷启动 ------------------------------------
if [[ "$STAGE_SFT" == "1" ]]; then
    echo ">>> [3/5] SFT 训练 ..."
    torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MAIN_PORT} \
        sft.py \
        --base_model ${MODEL_PATH} \
        --batch_size 256 \
        --micro_batch_size 16 \
        --train_file ${train_file} \
        --eval_file ${valid_file} \
        --output_dir ${SFT_OUTPUT_DIR} \
        --wandb_project edu_rec \
        --wandb_run_name edu_sft \
        --category ${CATEGORY} \
        --train_from_scratch False \
        --seed ${SEED} \
        --sid_index_path ${SID_INDEX_PATH} \
        --item_meta_path ${ITEM_META_PATH} \
        --freeze_LLM False
    echo ">>> SFT 完成 -> ${SFT_OUTPUT_DIR}"
fi

# --------------------------- ④ RL (ZPD reward) -------------------------------
if [[ "$STAGE_RL" == "1" ]]; then
    echo ">>> [4/5] RL 训练 (reward=${REWARD_TYPE}) ..."
    # RL 基座：优先用 SFT 产物
    RL_BASE=${RL_BASE:-"${SFT_OUTPUT_DIR}/final_checkpoint"}
    [[ -d "${RL_BASE}" ]] || RL_BASE=${MODEL_PATH}
    echo "    RL base model: ${RL_BASE}"

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes ${NUM_GPUS} --main_process_port ${MAIN_PORT} \
        rl.py \
        --model_path ${RL_BASE} \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file ${train_file} \
        --eval_file ${valid_file} \
        --info_file ${info_file} \
        --category ${CATEGORY} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ${REWARD_TYPE} \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir ${RL_OUTPUT_DIR} \
        --wandb_run_name edu_rl \
        --sid_index_path ${SID_INDEX_PATH} \
        --item_meta_path ${ITEM_META_PATH}
    echo ">>> RL 完成 -> ${RL_OUTPUT_DIR}"
fi

# --------------------------- ⑤ 评估 ------------------------------------------
if [[ "$STAGE_EVAL" == "1" ]]; then
    echo ">>> [5/5] 评估 ..."
    EVAL_MODEL=${EVAL_MODEL:-"${RL_OUTPUT_DIR}/final_checkpoint"}
    [[ -d "${EVAL_MODEL}" ]] || EVAL_MODEL=${RL_OUTPUT_DIR}
    [[ -d "${EVAL_MODEL}" ]] || EVAL_MODEL=${SFT_OUTPUT_DIR}/final_checkpoint
    echo "    eval model: ${EVAL_MODEL}"

    exp_name_clean=$(basename "${EVAL_MODEL}")
    result_dir="./results/edu_${exp_name_clean}"
    mkdir -p "${result_dir}"
    result_json="${result_dir}/final_result_${CATEGORY}.json"

    CUDA_VISIBLE_DEVICES=0 python -u ./evaluate.py \
        --base_model "${EVAL_MODEL}" \
        --info_file "${info_file}" \
        --category ${CATEGORY} \
        --test_data_path "${test_file}" \
        --result_json_data "${result_json}" \
        --batch_size 8 \
        --num_beams 50 \
        --max_new_tokens 256 \
        --temperature 1.0 \
        --guidance_scale 1.0 \
        --length_penalty 0.0

    echo ">>> 计算指标 ..."
    python ./calc.py --path "${result_json}" --item_path "${info_file}"
    echo ">>> 评估完成 -> ${result_json}"
fi

echo "=============================================================="
echo " 全链路结束！"
echo "=============================================================="
