# -*- coding: utf-8 -*-
"""
教育「题目级」推荐数据自检脚本（不依赖模型）。

校验维度：
  A. Item / SID 层（基于 raw item.json + index.json）
     - SID 格式合法性：全部匹配 <a_x><b_y><c_z>
     - SID 唯一性：无两题共享同一 SID
     - 三层结构统计：知识点(a)数、各 a 下难度档(b)、题序(c)
     - 难度分布：difficulty 的 min/max/mean + 分桶
     - ZPD 单调性：同一知识点内，难度档(b)越大 difficulty 不应下降
  B. 交互层（基于 .train/.valid/.test.inter）
     - 样本数、序列长度分布（min/max/mean/分位）
     - 每题覆盖度：作为 target 被练习的题数、零覆盖题数
     - 自穿越：target 出现在自身 history 中的样本数
     - 跨 split 泄漏：同一 (history, target) 样本重复出现在多个 split
  C. CSV ↔ raw 一致性（基于 data/Edu/{train,valid,test}/*.csv）
     - item_sid 与 index.json 拼接一致
     - item_title 与 item.json 一致
     - history_item_sid 与 history_item_id 长度一致

用法：
  python data/edu_data_check.py \
      --raw_dir data/Edu_raw/Edu_Questions \
      --csv_dir data/Edu \
      --dataset_name Edu_Questions

发现严重问题（FAIL）时返回非零退出码，便于在流水线里卡住。
"""

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

# Windows 控制台默认 GBK，无法打印部分符号；强制 stdout 用 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

SID_RE = re.compile(r"^<a_(\d+)><b_(\d+)><c_(\d+)>$")

# ----------------------------- 工具 ----------------------------- #

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results = []  # (level, title, detail)


def record(level, title, detail=""):
    _results.append((level, title, detail))
    tag = {PASS: "[ PASS ]", WARN: "[ WARN ]", FAIL: "[ FAIL ]"}[level]
    line = f"{tag} {title}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def section(name):
    print("\n" + "=" * 64)
    print(f" {name}")
    print("=" * 64)


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


# ----------------------------- 加载 ----------------------------- #

def load_raw(raw_dir, name):
    item_path = os.path.join(raw_dir, f"{name}.item.json")
    index_path = os.path.join(raw_dir, f"{name}.index.json")
    with open(item_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return items, index


def load_inter(raw_dir, name, split):
    """读 .inter，返回样本列表 [(history:list[int], target:int)]。"""
    path = os.path.join(raw_dir, f"{name}.{split}.inter")
    if not os.path.exists(path):
        return None
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            hist = [int(x) for x in parts[1].split()] if parts[1].strip() else []
            tgt = int(parts[2])
            samples.append((hist, tgt))
    return samples


# ----------------------------- A. SID 层 ----------------------------- #

def check_items(items, index):
    section("A. Item / SID 层")

    n_items = len(items)
    n_index = len(index)
    if n_items == n_index:
        record(PASS, f"题目数量一致：item={n_items}, index={n_index}")
    else:
        record(FAIL, "item.json 与 index.json 数量不一致",
               f"item={n_items}, index={n_index}")

    # SID 拼接 + 格式合法性
    sid_of = {}
    bad_fmt = []
    a_set, ab_set, abc_dup = set(), set(), Counter()
    for iid, parts in index.items():
        sid = "".join(parts)
        sid_of[iid] = sid
        m = SID_RE.match(sid)
        if not m:
            bad_fmt.append((iid, sid))
            continue
        a, b, c = m.group(1), m.group(2), m.group(3)
        a_set.add(a)
        ab_set.add((a, b))
        abc_dup[sid] += 1

    if bad_fmt:
        record(FAIL, f"存在非法 SID 格式：{len(bad_fmt)} 个",
               f"示例：{bad_fmt[:3]}")
    else:
        record(PASS, "全部 SID 格式合法（<a_x><b_y><c_z>）")

    # SID 唯一性
    dups = {s: c for s, c in abc_dup.items() if c > 1}
    if dups:
        sample = list(dups.items())[:3]
        record(FAIL, f"存在重复 SID：{len(dups)} 个被多题共用",
               f"示例：{sample}")
    else:
        record(PASS, f"SID 唯一性 OK（{len(abc_dup)} 个唯一 SID）")

    # 三层结构统计
    b_per_a = defaultdict(set)
    c_per_ab = defaultdict(set)
    for iid, parts in index.items():
        m = SID_RE.match("".join(parts))
        if not m:
            continue
        a, b, c = m.groups()
        b_per_a[a].add(b)
        c_per_ab[(a, b)].add(c)
    n_kp = len(a_set)
    avg_b = sum(len(v) for v in b_per_a.values()) / max(n_kp, 1)
    avg_c = sum(len(v) for v in c_per_ab.values()) / max(len(ab_set), 1)
    record(PASS, "三层结构统计",
           f"知识点(a)={n_kp}，平均难度档(b)/知识点={avg_b:.2f}，平均题序(c)/(a,b)={avg_c:.2f}")

    # difficulty 分布
    diffs = [v.get("difficulty") for v in items.values() if isinstance(v, dict) and "difficulty" in v]
    diffs = [d for d in diffs if d is not None]
    if diffs:
        sd = sorted(diffs)
        buckets = Counter()
        for d in diffs:
            buckets[min(int(d * 5), 4)] = buckets.get(min(int(d * 5), 4), 0) + 1
        bk = ", ".join(f"[{i*0.2:.1f}-{(i+1)*0.2:.1f}):{buckets.get(i,0)}" for i in range(5))
        record(PASS, "难度系数分布",
               f"min={sd[0]:.2f}, mean={sum(sd)/len(sd):.2f}, max={sd[-1]:.2f}；分桶 {bk}")
    else:
        record(WARN, "item.json 缺少 difficulty 字段，跳过难度分布")

    # ZPD 单调性：同一知识点(a)内，b 越大 difficulty 不应下降
    # 取每个 (a,b) 的平均 difficulty，检查随 b 单调不减
    ab_diff = defaultdict(list)
    for iid, v in items.items():
        if iid not in index:
            continue
        m = SID_RE.match("".join(index[iid]))
        if not m or not isinstance(v, dict) or "difficulty" not in v:
            continue
        a, b, _ = m.groups()
        ab_diff[(a, int(b))].append(v["difficulty"])
    violations = 0
    checked_kp = 0
    for a in a_set:
        seq = sorted([(b, sum(ds) / len(ds)) for (aa, b), ds in ab_diff.items() if aa == a])
        if len(seq) < 2:
            continue
        checked_kp += 1
        for i in range(1, len(seq)):
            if seq[i][1] < seq[i - 1][1] - 1e-9:
                violations += 1
                break
    if checked_kp == 0:
        record(WARN, "无足够数据校验 ZPD 难度单调性")
    elif violations == 0:
        record(PASS, f"ZPD 难度单调性 OK（{checked_kp} 个知识点，难度档越高难度系数不下降）")
    else:
        record(WARN, f"ZPD 难度单调性存在反例：{violations}/{checked_kp} 个知识点",
               "难度档(b)增大时平均难度系数下降，可能影响 ZPD reward 区分度")

    return sid_of


# ----------------------------- B. 交互层 ----------------------------- #

def check_inter(raw_dir, name, items):
    section("B. 交互层（序列 / 覆盖度 / 穿越 / 泄漏）")

    all_item_ids = set(int(i) for i in items.keys())
    split_samples = {}
    for split in ("train", "valid", "test"):
        s = load_inter(raw_dir, name, split)
        if s is None:
            record(WARN, f"缺少 {name}.{split}.inter，跳过")
            continue
        split_samples[split] = s

    if not split_samples:
        record(FAIL, "未找到任何 inter 文件")
        return

    for split, samples in split_samples.items():
        lens = sorted(len(h) for h, _ in samples)
        n = len(samples)
        self_leak = sum(1 for h, t in samples if t in set(h))
        oob = sum(1 for h, t in samples
                  if t not in all_item_ids or any(x not in all_item_ids for x in h))

        detail = (f"样本={n}；history长度 min={lens[0]}, "
                  f"p50={percentile(lens,0.5)}, p90={percentile(lens,0.9)}, max={lens[-1]}")
        record(PASS, f"[{split}] 序列长度分布", detail)

        if self_leak == 0:
            record(PASS, f"[{split}] 无自穿越（target 不在自身 history 中）")
        else:
            record(WARN, f"[{split}] 自穿越样本：{self_leak}/{n}",
                   "target 出现在自身 history 中（教育场景可能为重复练习，按需判断）")

        if oob == 0:
            record(PASS, f"[{split}] 所有 item_id 均在题库内")
        else:
            record(FAIL, f"[{split}] 越界 item_id 样本：{oob}/{n}",
                   "存在题库中不存在的 item_id")

    # 每题覆盖度（基于 train 的 target）
    if "train" in split_samples:
        tgt_cnt = Counter(t for _, t in split_samples["train"])
        covered = len(tgt_cnt)
        zero = len(all_item_ids) - covered
        cov_rate = covered / max(len(all_item_ids), 1)
        cvals = sorted(tgt_cnt.values())
        detail = (f"被练习题数={covered}/{len(all_item_ids)}（{cov_rate*100:.1f}%），"
                  f"零覆盖题={zero}；每题被练 min={cvals[0]}, "
                  f"p50={percentile(cvals,0.5)}, max={cvals[-1]}")
        if zero == 0:
            record(PASS, "[train] 每题覆盖度（无零覆盖题）", detail)
        elif cov_rate >= 0.9:
            record(WARN, "[train] 存在少量零覆盖题", detail)
        else:
            record(FAIL, "[train] 覆盖度偏低，冷门题学不动", detail)

    # 跨 split 泄漏：完全相同的 (history, target) 出现在不同 split
    sig = {}
    for split, samples in split_samples.items():
        sig[split] = set((tuple(h), t) for h, t in samples)
    splits = list(sig.keys())
    leak_found = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = sig[splits[i]] & sig[splits[j]]
            if inter:
                leak_found = True
                record(WARN, f"跨 split 重复样本：{splits[i]} ∩ {splits[j]} = {len(inter)}",
                       "完全相同的 (history,target) 同时出现在两个划分")
    if not leak_found:
        record(PASS, "跨 split 无完全相同的 (history,target) 样本")


# ----------------------------- C. CSV 一致性 ----------------------------- #

def check_csv(csv_dir, items, index):
    section("C. CSV <-> raw 一致性")

    sid_of = {int(i): "".join(p) for i, p in index.items()}
    title_of = {int(i): v.get("title") for i, v in items.items() if isinstance(v, dict)}

    total = 0
    sid_mismatch = 0
    title_mismatch = 0
    len_mismatch = 0
    checked_any = False

    for split in ("train", "valid", "test"):
        path = find_one(os.path.join(csv_dir, split, "*.csv"))
        if not path:
            record(WARN, f"未找到 {split}/*.csv，跳过")
            continue
        checked_any = True
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                try:
                    iid = int(row["item_id"])
                except (KeyError, ValueError):
                    continue
                # item_sid 一致
                if row.get("item_sid") != sid_of.get(iid):
                    sid_mismatch += 1
                # item_title 一致
                if title_of.get(iid) is not None and row.get("item_title") != title_of.get(iid):
                    title_mismatch += 1
                # history 长度一致
                try:
                    h_ids = ast.literal_eval(row.get("history_item_id", "[]"))
                    h_sids = ast.literal_eval(row.get("history_item_sid", "[]"))
                    if len(h_ids) != len(h_sids):
                        len_mismatch += 1
                except (ValueError, SyntaxError):
                    pass

    if not checked_any:
        record(WARN, "未找到任何 CSV，跳过一致性校验")
        return

    record(PASS, f"已校验 CSV 行数：{total}")
    for cnt, ok_msg, bad_title in [
        (sid_mismatch, "item_sid 与 index.json 完全一致", "item_sid 与 index.json 不一致"),
        (title_mismatch, "item_title 与 item.json 完全一致", "item_title 与 item.json 不一致"),
        (len_mismatch, "history 的 id 与 sid 长度一致", "history 的 id 与 sid 长度不一致"),
    ]:
        if cnt == 0:
            record(PASS, ok_msg)
        else:
            record(FAIL, f"{bad_title}：{cnt} 行")


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser(description="教育题目级推荐数据自检（不依赖模型）")
    ap.add_argument("--raw_dir", default="data/Edu_raw/Edu_Questions",
                    help="raw 数据目录（含 *.item.json / *.index.json / *.inter）")
    ap.add_argument("--csv_dir", default="data/Edu",
                    help="转换后 CSV 根目录（含 train/ valid/ test/）")
    ap.add_argument("--dataset_name", default="Edu_Questions")
    args = ap.parse_args()

    print("教育题目级推荐 · 数据自检报告")
    print(f"raw_dir={args.raw_dir}  csv_dir={args.csv_dir}  dataset={args.dataset_name}")

    items, index = load_raw(args.raw_dir, args.dataset_name)
    check_items(items, index)
    check_inter(args.raw_dir, args.dataset_name, items)
    check_csv(args.csv_dir, items, index)

    # 汇总
    section("汇总")
    n_pass = sum(1 for lv, _, _ in _results if lv == PASS)
    n_warn = sum(1 for lv, _, _ in _results if lv == WARN)
    n_fail = sum(1 for lv, _, _ in _results if lv == FAIL)
    print(f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")
    if n_fail:
        print("\n结论：存在 FAIL 项，建议修复后再训练。")
        sys.exit(1)
    elif n_warn:
        print("\n结论：无致命问题，但有 WARN 项，请按业务语义确认。")
        sys.exit(0)
    else:
        print("\n结论：全部通过，数据可用于训练。")
        sys.exit(0)


if __name__ == "__main__":
    main()
