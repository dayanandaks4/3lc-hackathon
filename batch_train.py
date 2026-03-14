"""
Batch training sweep for 3LC x Sphere Hive Hackathon.

Behavior:
1) Loads labeled rows from train-autolabeled-v7.
2) Loads the original labeled seed rows from train (expected ~100 labeled).
3) Splits additional labeled rows into batches of 10.
4) For each batch, creates a new 3LC table with (seed 100 + batch).
5) Runs train.py on each batch table and records validation accuracy.
6) Prints a summary and best batch.

Notes:
- To fit hackathon time constraints, this script defaults to EPOCHS=1 per batch.
  Override with --epochs-per-batch if needed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import tlc

PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
BASE_TABLE_URL = (
    "/home/jocky/.local/share/3LC/projects/Chihuahua-Muffin/"
    "datasets/chihuahua-muffin/tables/train"
)
AUTO_TABLE_URL = (
    "/home/jocky/.local/share/3LC/projects/Chihuahua-Muffin/"
    "datasets/chihuahua-muffin/tables/train-autolabeled-v7"
)

SCHEMAS = {
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=["chihuahua", "muffin", "undefined"]),
    "weight": tlc.SampleWeightSchema(),
}

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "batch_train_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-of-10 train-table sweep.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for additional labeled rows.")
    parser.add_argument("--epochs-per-batch", type=int, default=1, help="Training epochs for each batch run.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap for debugging (0 means run all batches).",
    )
    return parser.parse_args()


def extract_best_accuracy(text: str) -> float | None:
    m = re.search(r"Best validation accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    return float(m.group(1)) if m else None


def split_batches(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    return [df.iloc[i : i + n].copy() for i in range(0, len(df), n)]


def build_table(table_name: str, rows_df: pd.DataFrame) -> str:
    writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Batch table {table_name} (seed+batch)",
        column_schemas=SCHEMAS,
        if_exists="overwrite",
    )

    for r in rows_df.itertuples(index=False):
        writer.add_row(
            {
                "image": str(r.image),
                "label": int(r.label),
                "weight": float(r.weight),
            }
        )

    table = writer.finalize()
    return str(table.url)


def run_training(table_name: str, epochs: int, log_path: Path) -> tuple[int, float | None]:
    env = os.environ.copy()
    env["TRAIN_TABLE_NAME"] = table_name
    env["EPOCHS"] = str(epochs)

    cmd = [sys.executable, str(ROOT / "train.py")]
    with log_path.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True).returncode

    text = log_path.read_text(encoding="utf-8", errors="replace")
    acc = extract_best_accuracy(text)
    return rc, acc


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("  Batch training sweep")
    print("=" * 72)

    base_table = tlc.Table.from_url(BASE_TABLE_URL)
    auto_table = tlc.Table.from_url(AUTO_TABLE_URL)

    base_df = base_table.to_pandas()
    auto_df = auto_table.to_pandas()

    base_labeled = base_df[base_df["label"] != 2].copy()
    auto_labeled = auto_df[auto_df["label"] != 2].copy()

    base_labeled["weight"] = 1.0
    auto_labeled["weight"] = 1.0

    seed_images = set(base_labeled["image"].astype(str).tolist())
    add_df = auto_labeled[~auto_labeled["image"].astype(str).isin(seed_images)].copy()

    print(f"Seed labeled rows (from train): {len(base_labeled)}")
    print(f"All labeled rows (v7):          {len(auto_labeled)}")
    print(f"Additional labeled rows:         {len(add_df)}")

    batches = split_batches(add_df, args.batch_size)
    if args.max_batches > 0:
        batches = batches[: args.max_batches]
    print(f"Total batches to run:            {len(batches)}")
    print(f"Batch size:                      {args.batch_size}")
    print(f"Epochs per batch run:            {args.epochs_per_batch}")

    if not batches:
        print("[ERROR] No batches to run.")
        return 1

    results: list[tuple[int, str, float]] = []

    for i, bdf in enumerate(batches, start=1):
        table_name = f"train-batch-v7-{i:03d}"
        run_df = pd.concat([base_labeled, bdf], ignore_index=True)

        table_url = build_table(table_name, run_df)
        print(f"\n[{i}/{len(batches)}] table={table_name} rows={len(run_df)}")
        print(f"  URL: {table_url}")

        log_path = LOG_DIR / f"batch_{i:03d}.log"
        rc, acc = run_training(table_name, args.epochs_per_batch, log_path)

        if rc != 0:
            print(f"  [FAIL] train.py exit={rc} (log: {log_path})")
            continue
        if acc is None:
            print(f"  [FAIL] Could not parse accuracy (log: {log_path})")
            continue

        print(f"  [OK] Best validation accuracy: {acc:.2f}%")
        results.append((i, table_name, acc))

    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)

    if not results:
        print("No successful batch runs.")
        return 1

    for i, name, acc in results:
        print(f"Batch {i:03d} | {name} | {acc:.2f}%")

    best_i, best_name, best_acc = sorted(results, key=lambda x: x[2], reverse=True)[0]
    print("\nBest batch:")
    print(f"  Batch index: {best_i:03d}")
    print(f"  Table name:  {best_name}")
    print(f"  Accuracy:    {best_acc:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
