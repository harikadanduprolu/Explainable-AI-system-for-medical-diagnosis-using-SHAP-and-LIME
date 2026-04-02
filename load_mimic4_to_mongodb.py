#!/usr/bin/env python3
"""
Load local MIMIC-IV CSV.gz tables into MongoDB.

Default behavior imports a practical subset (patients, admissions, icustays)
from the local PhysioNet directory in chunked batches.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError


DEFAULT_MIMIC_ROOT = Path("dataset/mimic4/physionet.org/files/mimiciv/3.1")
DEFAULT_TABLES = ["patients", "admissions", "icustays"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MIMIC-IV CSV.gz data into MongoDB")
    parser.add_argument(
        "--mimic-root",
        type=str,
        default=str(DEFAULT_MIMIC_ROOT),
        help="Path to MIMIC-IV 3.1 root folder containing hosp/ and icu/",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=os.getenv("MONGODB_URL"),
        help="MongoDB URI (defaults to env MONGODB_URL)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=os.getenv("DATABASE_NAME", "medical_ai_db"),
        help="MongoDB database name",
    )
    parser.add_argument(
        "--tables",
        type=str,
        default=",".join(DEFAULT_TABLES),
        help="Comma-separated tables to import (e.g. patients,admissions,icustays)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Insert batch size",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=None,
        help="Optional row cap per table for test runs",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop destination collection before import",
    )
    return parser.parse_args()


def resolve_table_file(mimic_root: Path, table: str) -> Path:
    hosp_file = mimic_root / "hosp" / f"{table}.csv.gz"
    icu_file = mimic_root / "icu" / f"{table}.csv.gz"

    if hosp_file.exists():
        return hosp_file
    if icu_file.exists():
        return icu_file
    raise FileNotFoundError(f"Could not find table '{table}' under hosp/ or icu/")


def convert_chunk(chunk: pd.DataFrame) -> List[Dict]:
    # Convert NaN/NaT to None so MongoDB stores nulls cleanly.
    cleaned = chunk.where(pd.notna(chunk), None)
    return cleaned.to_dict(orient="records")


def create_common_indexes(collection, sample_columns: List[str]) -> None:
    candidates = ["subject_id", "hadm_id", "stay_id", "charttime"]
    for col in candidates:
        if col in sample_columns:
            collection.create_index(col)


def import_table(
    db,
    mimic_root: Path,
    table: str,
    batch_size: int,
    max_rows_per_table: Optional[int],
    drop_existing: bool,
) -> int:
    table_file = resolve_table_file(mimic_root, table)
    collection_name = f"mimic4_{table}"
    collection = db[collection_name]

    if drop_existing:
        collection.drop()

    print(f"\nImporting {table} from {table_file} -> {collection_name}")

    inserted_total = 0
    first_chunk = True

    for chunk in pd.read_csv(table_file, compression="gzip", chunksize=batch_size, low_memory=False):
        if max_rows_per_table is not None:
            remaining = max_rows_per_table - inserted_total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        records = convert_chunk(chunk)
        if not records:
            continue

        if first_chunk:
            create_common_indexes(collection, list(chunk.columns))
            first_chunk = False

        try:
            collection.insert_many(records, ordered=False)
            inserted_total += len(records)
        except BulkWriteError as exc:
            # Continue on duplicate key or partial insert errors.
            n_inserted = int(exc.details.get("nInserted", 0))
            inserted_total += n_inserted

        if inserted_total % (batch_size * 5) == 0:
            print(f"  inserted {inserted_total} rows...")

    print(f"Done {table}: inserted {inserted_total} rows")
    return inserted_total


def main() -> None:
    args = parse_args()

    if not args.mongo_uri:
        raise ValueError("MongoDB URI is required. Set MONGODB_URL or pass --mongo-uri")

    mimic_root = Path(args.mimic_root)
    if not mimic_root.exists():
        raise FileNotFoundError(f"MIMIC root not found: {mimic_root}")

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    if not tables:
        raise ValueError("No tables selected")

    client = MongoClient(args.mongo_uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    db = client[args.database]

    print(f"Connected to MongoDB. Database: {args.database}")
    print(f"MIMIC root: {mimic_root}")
    print(f"Tables: {tables}")

    summary = {}
    for table in tables:
        inserted = import_table(
            db=db,
            mimic_root=mimic_root,
            table=table,
            batch_size=args.batch_size,
            max_rows_per_table=args.max_rows_per_table,
            drop_existing=args.drop_existing,
        )
        summary[table] = inserted

    print("\nImport summary:")
    for table, inserted in summary.items():
        print(f"  {table}: {inserted}")


if __name__ == "__main__":
    main()
