#!/usr/bin/env python3
"""
Create smaller train/dev CSV manifests from the full dataset manifests.

This keeps the original clips and manifests untouched. It only writes
new CSVs under data/ for a faster first-pass training run.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(rows)


def source_key(row):
    path = Path(row["path"])
    parts = path.parts
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "raw":
        return parts[2]
    return path.parent.name


def proportional_sample(rows, target_count, seed):
    if target_count >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[source_key(row)].append(row)

    sampled = []
    total = len(rows)
    remaining = target_count
    group_items = list(groups.items())

    # First pass: proportional allocation with a minimum of 1 per group.
    allocations = {}
    for idx, (group, items) in enumerate(group_items):
        if idx == len(group_items) - 1:
            alloc = remaining
        else:
            alloc = max(1, round(target_count * len(items) / total))
            alloc = min(alloc, len(items), remaining - (len(group_items) - idx - 1))
        allocations[group] = alloc
        remaining -= alloc

    # Second pass: adjust in case rounding exceeded or undershot.
    current_total = sum(allocations.values())
    while current_total > target_count:
        for group, items in group_items:
            if allocations[group] > 1 and current_total > target_count:
                allocations[group] -= 1
                current_total -= 1
    while current_total < target_count:
        for group, items in group_items:
            if allocations[group] < len(items) and current_total < target_count:
                allocations[group] += 1
                current_total += 1

    for idx, (group, items) in enumerate(group_items):
        rng.shuffle(items)
        sampled.extend(items[:allocations[group]])

    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", default="data/transcripts_train.csv")
    parser.add_argument("--dev-in", default="data/transcripts_dev.csv")
    parser.add_argument("--train-out", default="data/transcripts_train_subset.csv")
    parser.add_argument("--dev-out", default="data/transcripts_dev_subset.csv")
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--dev-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = read_rows(Path(args.train_in))
    dev_rows = read_rows(Path(args.dev_in))

    train_subset = proportional_sample(train_rows, args.train_count, args.seed)
    dev_subset = proportional_sample(dev_rows, args.dev_count, args.seed + 1)

    write_rows(Path(args.train_out), train_subset)
    write_rows(Path(args.dev_out), dev_subset)

    print(f"Wrote {len(train_subset)} rows -> {args.train_out}")
    print(f"Wrote {len(dev_subset)} rows -> {args.dev_out}")


if __name__ == "__main__":
    main()
