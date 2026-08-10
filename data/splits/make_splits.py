"""Split the labeled NPC cohort (data/labels/Case*.json) into train/valid/test
sets, at the patient level (each patient has exactly one case here).

Produces 6 CSVs in data/splits/:
  - random_train.csv, random_valid.csv, random_test.csv
      Plain random split, no regard to outcome balance.
  - stratified_train.csv, stratified_valid.csv, stratified_test.csv
      Split within each event group (death vs censored) at the same
      ratio, so all three sets keep the overall event/censored ratio.

Only cases with a real per-case label file are included (the aggregate
data/labels/pfs_survival_groundtruth.json is not a per-case file and is
skipped); imaged cases with no outcome label are excluded.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

CASE_FILE_RE = re.compile(r"^Case\d{3}\.json$")
FIELDS = ["case_id", "patient_id", "event", "time_months"]


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(LABEL_DIR.glob("*.json")):
        if not CASE_FILE_RE.match(path.name):
            continue
        with open(path) as f:
            d = json.load(f)
        cases.append({k: d[k] for k in FIELDS})
    return cases


def ratio_split(items: list, ratios: tuple[float, float, float]) -> tuple[list, list, list]:
    n = len(items)
    n_train = round(n * ratios[0])
    n_valid = round(n * ratios[1])
    n_train = min(n_train, n)
    n_valid = min(n_valid, n - n_train)
    n_test = n - n_train - n_valid
    return items[:n_train], items[n_train:n_train + n_valid], items[n_train + n_valid:]


def random_split(cases: list[dict], ratios: tuple[float, float, float], rng: random.Random):
    shuffled = cases[:]
    rng.shuffle(shuffled)
    return ratio_split(shuffled, ratios)


def stratified_split(cases: list[dict], ratios: tuple[float, float, float], rng: random.Random):
    by_event: dict = {}
    for c in cases:
        by_event.setdefault(c["event"], []).append(c)

    train, valid, test = [], [], []
    for group in by_event.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        g_train, g_valid, g_test = ratio_split(shuffled, ratios)
        train += g_train
        valid += g_valid
        test += g_test

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test) 
    return train, valid, test


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(name: str, train: list[dict], valid: list[dict], test: list[dict]):
    def event_rate(rows):
        return sum(r["event"] for r in rows) / len(rows) if rows else 0.0

    print(f"[{name}] train={len(train)} (event={event_rate(train):.3f}) "
          f"valid={len(valid)} (event={event_rate(valid):.3f}) "
          f"test={len(test)} (event={event_rate(test):.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    cases = load_cases()
    print(f"Loaded {len(cases)} labeled cases")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    r_train, r_valid, r_test = random_split(cases, ratios, rng)
    summarize("random", r_train, r_valid, r_test)
    write_csv(OUTPUT_DIR / "random_train.csv", r_train)
    write_csv(OUTPUT_DIR / "random_valid.csv", r_valid)
    write_csv(OUTPUT_DIR / "random_test.csv", r_test)

    rng = random.Random(args.seed)
    s_train, s_valid, s_test = stratified_split(cases, ratios, rng)
    summarize("stratified", s_train, s_valid, s_test)
    write_csv(OUTPUT_DIR / "stratified_train.csv", s_train)
    write_csv(OUTPUT_DIR / "stratified_valid.csv", s_valid)
    write_csv(OUTPUT_DIR / "stratified_test.csv", s_test)


if __name__ == "__main__":
    main()
