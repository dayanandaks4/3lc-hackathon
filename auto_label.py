"""
Auto-label undefined train samples using 3LC run metrics.

What this script does:
1. Loads the latest train table for project Chihuahua-Muffin / dataset chihuahua-muffin
   via tlc.Table.from_url(...).latest().
2. Finds undefined-labeled rows in the train table.
3. Locates the newest run metrics table that still contains per-sample
   prediction columns (predicted, confidence, example_id).
4. Auto-labels undefined rows where confidence > threshold as predicted
   class (chihuahua/muffin), and sets weight to 1.
5. Writes an updated train table back to 3LC.

Usage:
    python auto_label.py
    python auto_label.py --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import tlc


PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
TABLE_NAME = "train"
DEFAULT_THRESHOLD = 0.90


@dataclass
class MetricsSource:
    run_name: str
    run_url: str
    metrics_table_url: str
    df: pd.DataFrame


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Some saved timestamps include a trailing Z
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_latest_train_table() -> tlc.Table:
    ref = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=TABLE_NAME,
    )
    # Explicitly use from_url(...).latest() as requested.
    table = tlc.Table.from_url(str(ref.url)).latest(wait_for_rescan=False)
    return table


def find_latest_metrics_with_predictions(project_name: str) -> MetricsSource:
    project_dir = Path.home() / ".local/share/3LC/projects" / project_name
    runs_dir = project_dir / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_json_paths = list(runs_dir.glob("*/object.3lc.json"))
    if not run_json_paths:
        raise RuntimeError("No run metadata found under project runs directory.")

    run_infos = []
    for run_json in run_json_paths:
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            last_modified = payload.get("last_modified") or payload.get("created")
            if not last_modified:
                continue
            run_infos.append((run_json, payload, _parse_dt(last_modified)))
        except Exception:
            # Ignore malformed run objects and keep scanning.
            continue

    if not run_infos:
        raise RuntimeError("Could not parse any run metadata files.")

    # Newest first.
    run_infos.sort(key=lambda x: x[2], reverse=True)

    required_cols = {"example_id", "predicted", "confidence"}

    for run_json, run_payload, _ in run_infos:
        run_dir = run_json.parent
        run_name = run_dir.name
        run_url = str(run_dir)

        for metric_info in run_payload.get("metrics", []):
            rel_metric_url = metric_info.get("url")
            if not rel_metric_url:
                continue

            metric_path = (run_dir / rel_metric_url).resolve()
            metric_url = str(metric_path)

            try:
                metric_table = tlc.Table.from_url(metric_url)
                metric_df = metric_table.to_pandas()
            except Exception:
                continue

            if required_cols.issubset(set(metric_df.columns)):
                return MetricsSource(
                    run_name=run_name,
                    run_url=run_url,
                    metrics_table_url=str(metric_table.url),
                    df=metric_df,
                )

    raise RuntimeError(
        "No recent run metrics table with columns "
        "{'example_id','predicted','confidence'} was found. "
        "Run train.py (with metrics collection) first."
    )


def _coerce_numeric(series: pd.Series, name: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().all():
        raise ValueError(f"Column '{name}' could not be converted to numeric values.")
    return coerced


def auto_label_undefined(
    train_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    label_map: dict[int, str],
    threshold: float,
) -> tuple[pd.DataFrame, int, int]:
    required_train_cols = {"id", "label", "weight"}
    missing_train = required_train_cols - set(train_df.columns)
    if missing_train:
        raise KeyError(f"Train table is missing required columns: {sorted(missing_train)}")

    required_metrics_cols = {"example_id", "predicted", "confidence"}
    missing_metrics = required_metrics_cols - set(metrics_df.columns)
    if missing_metrics:
        raise KeyError(f"Metrics table is missing required columns: {sorted(missing_metrics)}")

    label_map_lower = {k: v.lower() for k, v in label_map.items()}
    label_to_index = {v: k for k, v in label_map_lower.items()}
    undefined_idx = label_to_index.get("undefined", 2)

    updated = train_df.copy()

    updated["id"] = _coerce_numeric(updated["id"], "id").astype("Int64")
    updated["label"] = _coerce_numeric(updated["label"], "label").astype("Int64")
    updated["weight"] = _coerce_numeric(updated["weight"], "weight").astype(float)

    metrics = metrics_df[["example_id", "predicted", "confidence"]].copy()
    metrics["example_id"] = _coerce_numeric(metrics["example_id"], "example_id").astype("Int64")
    metrics["predicted"] = _coerce_numeric(metrics["predicted"], "predicted").astype("Int64")
    metrics["confidence"] = _coerce_numeric(metrics["confidence"], "confidence").astype(float)

    # Keep the last record if duplicates exist for a sample.
    metrics = metrics.dropna(subset=["example_id", "predicted", "confidence"])
    metrics = metrics.drop_duplicates(subset=["example_id"], keep="last")

    merged = updated.merge(
        metrics,
        how="left",
        left_on="id",
        right_on="example_id",
        suffixes=("", "_metric"),
    )

    was_undefined = merged["label"] == undefined_idx
    predicted_is_valid = merged["predicted"].isin([0, 1])
    high_conf = merged["confidence"] > threshold
    auto_label_mask = was_undefined & predicted_is_valid & high_conf

    undefined_total = int(was_undefined.sum())
    auto_labeled = int(auto_label_mask.sum())

    if auto_labeled > 0:
        merged.loc[auto_label_mask, "label"] = merged.loc[auto_label_mask, "predicted"].astype("Int64")
        merged.loc[auto_label_mask, "weight"] = 1.0

    # Keep original train schema columns only.
    out = merged[train_df.columns].copy()
    return out, undefined_total, auto_labeled


def save_table_revision(
    table_df: pd.DataFrame,
    table_name: str,
    threshold: float,
) -> tlc.Table:
    description = (
        "Auto-labeled undefined samples from latest run metrics "
        f"(confidence>{threshold:.2f})."
    )
    schemas = {
        "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
        "image": tlc.ImagePath,
        "label": tlc.CategoricalLabel("label", classes=["chihuahua", "muffin", "undefined"]),
        "weight": tlc.SampleWeightSchema(),
    }

    writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        column_schemas=schemas,
        description=description,
        if_exists="raise",
    )

    for row in table_df.itertuples(index=False):
        writer.add_row(
            {
                "id": int(row.id),
                "image": str(row.image),
                "label": int(row.label),
                "weight": float(row.weight),
            }
        )

    return writer.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label undefined train samples using latest 3LC metrics."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence threshold for auto-labeling (default: 0.90).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold = args.threshold

    if not 0.0 < threshold <= 1.0:
        print("[ERROR] --threshold must be in (0, 1].")
        return 2

    try:
        print("=" * 72)
        print("  Auto-label undefined samples from latest run metrics")
        print("=" * 72)

        train_table = load_latest_train_table()
        print(f"[OK] Loaded train table: {train_table.url}")

        label_map = train_table.get_simple_value_map("label")
        if not label_map:
            raise RuntimeError("Could not load label value-map from train table.")
        print(f"[OK] Label map: {label_map}")

        metrics_source = find_latest_metrics_with_predictions(PROJECT_NAME)
        print(f"[OK] Using run: {metrics_source.run_name}")
        print(f"[OK] Metrics table: {metrics_source.metrics_table_url}")

        train_df = train_table.to_pandas()
        if train_df.empty:
            raise RuntimeError("Train table is empty; nothing to auto-label.")

        updated_df, undefined_total, auto_labeled = auto_label_undefined(
            train_df=train_df,
            metrics_df=metrics_source.df,
            label_map=label_map,
            threshold=threshold,
        )

        print(f"[INFO] Undefined samples found: {undefined_total}")
        print(f"[INFO] Auto-labeled with confidence>{threshold:.2f}: {auto_labeled}")

        if auto_labeled == 0:
            print("[INFO] No rows met threshold. No new table revision written.")
            return 0

        # Find the first available table name: train-autolabeled-v1, v2, ...
        version = 1
        while True:
            candidate_table_name = f"train-autolabeled-v{version}"
            try:
                tlc.Table.from_names(
                    project_name=PROJECT_NAME,
                    dataset_name=DATASET_NAME,
                    table_name=candidate_table_name,
                ).latest(wait_for_rescan=False)
                version += 1
            except Exception:
                new_table_name = candidate_table_name
                break

        new_table = save_table_revision(
            table_df=updated_df,
            table_name=new_table_name,
            threshold=threshold,
        )

        latest_table = tlc.Table.from_names(
            project_name=PROJECT_NAME,
            dataset_name=DATASET_NAME,
            table_name=new_table_name,
        ).latest(wait_for_rescan=False)

        print("\n[OK] Wrote updated train table.")
        print(f"     New table URL: {new_table.url}")
        print(f"     Latest URL:    {latest_table.url}")
        print("[OK] Done.")
        return 0

    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
