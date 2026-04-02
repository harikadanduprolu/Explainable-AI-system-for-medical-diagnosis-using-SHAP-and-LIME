#!/usr/bin/env python3
"""Create a compact MIMIC-IV replica with each file below a size cap."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a partial replica of MIMIC-IV with per-file size limits."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/mimic4"),
        help="Source MIMIC-IV root folder.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("dataset/mimic4_mini"),
        help="Target folder for the compact replica.",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=25.0,
        help="Maximum size in MB per output file.",
    )
    parser.add_argument(
        "--initial-rows",
        type=int,
        default=20000,
        help="Initial row cap for CSV.gz files before resizing.",
    )
    return parser.parse_args()


def copy_small_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_capped_csv_gz(src: Path, dst: Path, max_bytes: int, initial_rows: int) -> tuple[int, int]:
    rows_limit = max(initial_rows, 100)

    while True:
        if dst.exists():
            dst.unlink()

        written = 0
        first_chunk = True

        for chunk in pd.read_csv(src, compression="gzip", chunksize=5000, low_memory=False):
            remaining = rows_limit - written
            if remaining <= 0:
                break

            part = chunk if len(chunk) <= remaining else chunk.iloc[:remaining]

            part.to_csv(
                dst,
                mode="wt" if first_chunk else "at",
                header=first_chunk,
                index=False,
                compression="gzip",
            )
            written += len(part)
            first_chunk = False

            if len(part) < len(chunk):
                break

        if not dst.exists():
            pd.read_csv(src, compression="gzip", nrows=0).to_csv(
                dst, index=False, compression="gzip"
            )

        size = dst.stat().st_size
        if size <= max_bytes:
            return written, size

        if rows_limit <= 100:
            return written, size

        rows_limit = max(rows_limit // 2, 100)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    max_bytes = int(args.max_mb * 1024 * 1024)

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    summary = []

    for src in source.rglob("*"):
        rel = src.relative_to(source)
        dst = target / rel

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        if src.name.endswith(".csv.gz"):
            rows, size = write_capped_csv_gz(src, dst, max_bytes, args.initial_rows)
            summary.append((str(rel), rows, size, "csv.gz"))
            print(f"[csv.gz] {rel} -> {rows} rows, {size / (1024 * 1024):.2f} MB")
        else:
            copy_small_file(src, dst)
            size = dst.stat().st_size
            summary.append((str(rel), -1, size, "copy"))
            print(f"[copy]   {rel} -> {size / (1024 * 1024):.2f} MB")

    oversized = [item for item in summary if item[2] > max_bytes]

    print("\nFinished creating compact replica")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Files:  {len(summary)}")
    print(f"Max MB: {args.max_mb}")

    if oversized:
        print("\nFiles above cap:")
        for rel, rows, size, mode in oversized:
            print(f" - {rel}: {size / (1024 * 1024):.2f} MB ({mode}, rows={rows})")
    else:
        print("All output files are within the size limit.")


if __name__ == "__main__":
    main()
