#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edu_data_process.py
========================================================================
把 MiniOneRec 迁移到「自适应学习路径 / 智能出题（题目级）」场景的
**造数 demo**：直接合成一批题目级模拟数据，产出与 convert_dataset.py
完全对齐的三种输入文件，从而无需真实学习日志即可跑通全链路。

产出（写到 --out_dir/<dataset_name>/ 下）：
  1. {dataset}.item.json   题目 meta：{qid: {title, description, difficulty, knowledge_point}}
  2. {dataset}.index.json  题目 → 三层 Semantic ID：{qid: ["<a_X>", "<b_Y>", "<c_Z>"]}
  3. {dataset}.train.inter / .valid.inter / .test.inter
     交互序列，格式（制表符分隔，首行为表头，被 convert_dataset.py 跳过）：
         student_id \t q1 q2 q3 ... \t target_q

为什么可以跳过 RQ-VAE？
  造数阶段我们直接**合成层级 SID**，让其携带教育语义：
    第 1 层 <a_X>  ->  知识点 id（同知识点的题，第一层相同 => SID 前缀相近）
    第 2 层 <b_Y>  ->  难度档（难度分桶，体现难度递进）
    第 3 层 <c_Z>  ->  同 (知识点, 难度档) 下的题序号，保证 SID 唯一
  真实业务里把这一步换成 rq/text2emb + rq/rqvae 即可，下游完全不变。

学生行为模拟（贴近 ZPD / 最近发展区）：
  - 每个学生有一个能力值 ability ∈ [0,1]
  - 每一步优先练习「难度接近自己能力」的题（softmax 采样）
  - 做对概率 = sigmoid(k * (ability - difficulty))
  - 做对后能力小幅提升 => 序列呈现「由易到难、循序渐进」的轨迹
========================================================================
用法示例：
  python data/edu_data_process.py \
      --out_dir data/Edu_raw \
      --dataset_name Edu_Questions \
      --num_kp 30 --q_per_kp 8 \
      --num_students 800 --seed 42
"""

import os
import json
import math
import random
import argparse
from collections import defaultdict


# --------------------------------------------------------------------- #
# 题库：知识点 + 题目模板（纯造数，可自行替换为真实题库）
# --------------------------------------------------------------------- #
KNOWLEDGE_POINTS = [
    "有理数运算", "整式加减", "一元一次方程", "二元一次方程组", "不等式与不等式组",
    "平面直角坐标系", "一次函数", "整式乘法与因式分解", "分式", "二次根式",
    "一元二次方程", "二次函数", "反比例函数", "全等三角形", "相似三角形",
    "勾股定理", "四边形", "圆", "概率初步", "统计初步",
    "锐角三角函数", "投影与视图", "数轴与绝对值", "幂的运算", "方程应用题",
    "函数图象分析", "几何证明", "数据分析", "图形变换", "代数综合",
]

# 难度档位（第二层 SID 的语义锚点），值为该档难度中心
DIFF_BUCKETS = [
    ("入门", 0.15),
    ("基础", 0.35),
    ("进阶", 0.55),
    ("综合", 0.75),
    ("拔高", 0.90),
]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def build_question_bank(num_kp, q_per_kp, seed):
    """生成题库：每个知识点下若干道难度递增的题目。

    返回:
        items:   {qid(str): {title, description, difficulty, knowledge_point}}
        index:   {qid(str): ["<a_X>", "<b_Y>", "<c_Z>"]}
        kp2qids: {kp_id: [qid, ...]}
        qid2diff:{qid(str): difficulty}
    """
    rng = random.Random(seed)
    items, index, kp2qids, qid2diff = {}, {}, defaultdict(list), {}

    qid = 0
    # 知识点名：内置列表用完后自动补名（知识点31、知识点32 ...），解除 30 上限
    def _kp_name(kp_id):
        if kp_id < len(KNOWLEDGE_POINTS):
            return KNOWLEDGE_POINTS[kp_id]
        return f"知识点{kp_id + 1}"

    for kp_id in range(num_kp):
        kp_name = _kp_name(kp_id)
        # 同一 (知识点, 难度档) 下的题序号计数器，作为第三层 SID
        seq_counter = defaultdict(int)
        for j in range(q_per_kp):
            # 难度在 [0,1] 上均匀铺开并加噪声，体现知识点内由易到难
            base = (j + 0.5) / q_per_kp
            difficulty = min(0.98, max(0.05, base + rng.uniform(-0.06, 0.06)))

            # 难度档（第二层 SID 语义）
            bucket_idx = min(
                range(len(DIFF_BUCKETS)),
                key=lambda b: abs(DIFF_BUCKETS[b][1] - difficulty),
            )
            bucket_name = DIFF_BUCKETS[bucket_idx][0]

            # 第三层：同 (kp, bucket) 下的序号，保证 SID 唯一
            c = seq_counter[(kp_id, bucket_idx)]
            seq_counter[(kp_id, bucket_idx)] += 1

            qid_s = str(qid)
            title = f"{kp_name}·{bucket_name}练习{c + 1}"
            description = (
                f"考查【{kp_name}】知识点的{bucket_name}难度题目，"
                f"难度系数约 {difficulty:.2f}，建议在掌握前置知识点后练习。"
            )
            items[qid_s] = {
                "title": title,
                "description": description,
                "difficulty": round(difficulty, 4),
                "knowledge_point": kp_name,
            }
            # 三层语义 ID：知识点 / 难度档 / 题内序号
            index[qid_s] = [f"<a_{kp_id}>", f"<b_{bucket_idx}>", f"<c_{c}>"]
            qid2diff[qid_s] = difficulty
            kp2qids[kp_id].append(qid_s)
            qid += 1

    return items, index, dict(kp2qids), qid2diff


def simulate_student_sequence(rng, all_qids, qid2diff, kp2qids,
                              min_len, max_len, tau=0.12, learn_rate=0.04):
    """模拟单个学生的练习轨迹（ZPD 行为 + 能力增长）。"""
    ability = rng.uniform(0.1, 0.6)          # 初始能力
    seq_len = rng.randint(min_len, max_len)

    # 学生倾向集中在 2~4 个知识点上练习（更真实）
    focus_kps = rng.sample(list(kp2qids.keys()),
                           k=min(len(kp2qids), rng.randint(2, 4)))
    candidate_qids = [q for kp in focus_kps for q in kp2qids[kp]]

    seq, seen = [], set()
    attempts = 0
    while len(seq) < seq_len and attempts < seq_len * 6:
        attempts += 1
        # 按「难度接近当前能力」做 softmax 采样
        logits = [-abs(qid2diff[q] - ability) / tau for q in candidate_qids]
        m = max(logits)
        weights = [math.exp(l - m) for l in logits]
        pick = rng.choices(candidate_qids, weights=weights, k=1)[0]
        if pick in seen:
            continue
        seen.add(pick)
        seq.append(pick)
        # 答题结果影响能力：做对则能力提升
        p_correct = sigmoid(8.0 * (ability - qid2diff[pick]))
        if rng.random() < p_correct:
            ability = min(1.0, ability + learn_rate * (1 - ability))
    return seq


def write_inter(path, rows):
    """写 .inter 文件：首行表头（被 convert_dataset.py 跳过），其余为数据行。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("student_id\thistory_questions\ttarget_question\n")
        for student_id, hist, target in rows:
            f.write(f"{student_id}\t{' '.join(hist)}\t{target}\n")


def build_splits(students_seqs):
    """leave-one-out 切分：
       test  = (hist=seq[:-1], target=seq[-1])
       valid = (hist=seq[:-2], target=seq[-2])
       train = 多个前缀滑窗 (hist=seq[:i], target=seq[i]), i = 2..len-3
    """
    train, valid, test = [], [], []
    for sid, seq in students_seqs:
        if len(seq) < 4:
            continue
        test.append((sid, seq[:-1], seq[-1]))
        valid.append((sid, seq[:-2], seq[-2]))
        for i in range(2, len(seq) - 2):
            train.append((sid, seq[:i], seq[i]))
    return train, valid, test


def main():
    ap = argparse.ArgumentParser(
        description="教育（题目级）造数 demo —— 产出 MiniOneRec 全链路输入文件")
    ap.add_argument("--out_dir", type=str, default="data/Edu_raw",
                    help="原始数据输出根目录")
    ap.add_argument("--dataset_name", type=str, default="Edu_Questions",
                    help="数据集名（同时作为文件名前缀与 category）")
    ap.add_argument("--num_kp", type=int, default=30,
                    help="知识点数量（超过内置30个时自动补名 知识点N）")
    ap.add_argument("--q_per_kp", type=int, default=8, help="每个知识点的题目数")
    ap.add_argument("--num_students", type=int, default=800, help="模拟学生数")
    ap.add_argument("--min_len", type=int, default=6, help="学生序列最短长度")
    ap.add_argument("--max_len", type=int, default=16, help="学生序列最长长度")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = os.path.join(args.out_dir, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) 题库 + Semantic ID
    items, index, kp2qids, qid2diff = build_question_bank(
        args.num_kp, args.q_per_kp, args.seed)
    all_qids = list(items.keys())
    print(f"[题库] 知识点 {len(kp2qids)} 个，题目 {len(items)} 道")

    # 2) 学生轨迹
    students_seqs = []
    for s in range(args.num_students):
        seq = simulate_student_sequence(
            rng, all_qids, qid2diff, kp2qids, args.min_len, args.max_len)
        if len(seq) >= 4:
            students_seqs.append((f"stu{s:05d}", seq))
    print(f"[学生] 有效学生序列 {len(students_seqs)} 条")

    # 3) 切分
    train, valid, test = build_splits(students_seqs)
    print(f"[切分] train={len(train)}  valid={len(valid)}  test={len(test)}")

    # 4) 落盘
    ds = args.dataset_name
    with open(os.path.join(out_dir, f"{ds}.item.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"{ds}.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    write_inter(os.path.join(out_dir, f"{ds}.train.inter"), train)
    write_inter(os.path.join(out_dir, f"{ds}.valid.inter"), valid)
    write_inter(os.path.join(out_dir, f"{ds}.test.inter"), test)

    print(f"\n[完成] 已写入 {out_dir}/")
    print(f"  - {ds}.item.json   ({len(items)} 题)")
    print(f"  - {ds}.index.json  (三层 SID)")
    print(f"  - {ds}.train/valid/test.inter")
    # 抽样展示
    sample_qid = all_qids[0]
    print("\n[示例] 题目 0:")
    print("  meta :", json.dumps(items[sample_qid], ensure_ascii=False))
    print("  SID  :", "".join(index[sample_qid]))
    print("\n下一步：用 convert_dataset.py 把上述文件转成训练 CSV + info.txt")


if __name__ == "__main__":
    main()
