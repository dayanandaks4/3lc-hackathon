"""
Rebuild the Chihuahua-Muffin train table from disk to recover from corrupted lineage.

What this script does:
1. Reads images from:
   - data/train/chihuahua  -> label 0, weight 1.0
   - data/train/muffin     -> label 1, weight 1.0
   - data/train/undefined  -> label 2, weight 0.0
2. Recreates the 3LC train table with if_exists="overwrite".
3. Runs a post-write sanity check that loads the table and verifies no lineage cycle error.

Usage:
    python fix_tables.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tlc

PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"
TABLE_NAME = "train"
CLASSES = ["chihuahua", "muffin", "undefined"]

SCHEMAS = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
    "image": tlc.ImagePath,
    "label": tlc.CategoricalLabel("label", classes=CLASSES),
    "weight": tlc.SampleWeightSchema(),
}


def collect_train_rows(train_dir: Path) -> list[dict]:
    class_specs = [
        ("chihuahua", 0, 1.0),
        ("muffin", 1, 1.0),
        ("undefined", 2, 0.0),
    ]

    rows: list[dict] = []
    next_id = 0

    for class_name, label, weight in class_specs:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Missing folder: {class_dir}")
            continue

        class_files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            class_files.extend(sorted(class_dir.glob(pattern)))

        print(f"[INFO] {class_name:10s}: {len(class_files):5d} images")

        for img_path in class_files:
            rows.append(
                {
                    "id": next_id,
                    "image": str(img_path.resolve()),
                    "label": label,
                    "weight": weight,
                }
            )
            next_id += 1

    return rows


def write_train_table(rows: list[dict]) -> tlc.Table:
    writer = tlc.TableWriter(
        table_name=TABLE_NAME,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description="Rebuilt train table to replace corrupted lineage cycle.",
        column_schemas=SCHEMAS,
        if_exists="overwrite",
    )

    for row in rows:
        writer.add_row(row)

    return writer.finalize()


def verify_no_cycle(table_url: str) -> None:
    # This is the same kind of access path that typically fails when lineage has cycles.
    table = tlc.Table.from_url(table_url).latest(wait_for_rescan=False)
    df = table.to_pandas()

    required = {"id", "image", "label", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Sanity check failed, missing columns: {sorted(missing)}")

    print(f"[OK] Sanity check loaded table with {len(df)} rows")
    print("[OK] No lineage cycle detected during latest()/to_pandas() checks")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    train_dir = base_dir / "data" / "train"

    print("=" * 72)
    print("  Rebuild 3LC train table (cycle recovery)")
    print("=" * 72)

    if not train_dir.exists():
        print(f"[ERROR] Train directory not found: {train_dir}")
        return 1

    try:
        tlc.register_project_url_alias(
            token="CHIHUAHUA_MUFFIN_DATA",
            path=str(base_dir),
            project=PROJECT_NAME,
        )
        print(f"[OK] Registered project alias for {base_dir}")

        rows = collect_train_rows(train_dir)
        if not rows:
            print("[ERROR] No training images were found. Aborting.")
            return 1

        table = write_train_table(rows)
        print(f"[OK] Rebuilt train table at: {table.url}")

        verify_no_cycle(str(table.url))

        labeled = sum(1 for r in rows if r["label"] in (0, 1))
        undefined = sum(1 for r in rows if r["label"] == 2)
        print(f"[INFO] Labeled rows (weight=1):   {labeled}")
        print(f"[INFO] Undefined rows (weight=0): {undefined}")
        print("[OK] Done.")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
