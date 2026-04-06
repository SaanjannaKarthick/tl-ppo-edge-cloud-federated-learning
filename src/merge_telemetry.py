#!/usr/bin/env python3
"""
Merge per-client telemetry CSV files into a single dataset.

Research-grade CLI tool for reproducible experiments.

Example:
    python merge_telemetry.py \
        --input_dir scripts/data \
        --output_csv scripts/data/telemetry_merged.csv
"""

import argparse
import glob
import os
import sys
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge telemetry_client*.csv files into one dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing telemetry_client*.csv files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output path for merged CSV file",
    )
    parser.add_argument(
        "--strict_schema",
        action="store_true",
        help="Fail if column mismatch is detected",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort by timestamp_utc and client_id",
    )
    return parser.parse_args()


def load_files(input_dir):
    pattern = os.path.join(input_dir, "telemetry_client*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    print(f"[INFO] Found {len(files)} files")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    return files


def validate_and_load(files, strict_schema):
    dfs = []
    base_cols = None

    for fp in files:
        df = pd.read_csv(fp)
        cols = list(df.columns)

        if base_cols is None:
            base_cols = cols
        else:
            if cols != base_cols:
                msg = (
                    f"[ERROR] Column mismatch in {os.path.basename(fp)}\n"
                    f"Expected: {base_cols}\nGot:      {cols}"
                )
                if strict_schema:
                    raise ValueError(msg)
                else:
                    print("[WARN] " + msg)

        df["source_file"] = os.path.basename(fp)
        dfs.append(df)

    return dfs


def merge_data(dfs):
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def process_timestamps(df):
    if "timestamp_utc" not in df.columns:
        print("[WARN] timestamp_utc column not found")
        return df

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    bad_ts = df["timestamp_utc"].isna().sum()

    if bad_ts:
        print(f"[WARN] {bad_ts} rows have invalid timestamp_utc")

    return df


def sort_data(df):
    if "timestamp_utc" in df.columns and "client_id" in df.columns:
        df = df.sort_values(
            ["timestamp_utc", "client_id"], kind="mergesort"
        )
    else:
        print("[WARN] Skipping sort (missing timestamp_utc or client_id)")

    return df


def save_output(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DONE] Saved merged dataset: {output_path}")


def print_summary(df):
    print("\n[SUMMARY] Total rows:", len(df))

    if "client_id" in df.columns:
        print("\n[SUMMARY] Rows per client_id:")
        print(df["client_id"].value_counts().sort_index())

    if "scenario" in df.columns:
        print("\n[SUMMARY] Rows per scenario:")
        print(df["scenario"].value_counts())

    print("\n[SUMMARY] Columns:")
    print(list(df.columns))


def main():
    args = parse_args()

    try:
        files = load_files(args.input_dir)
        dfs = validate_and_load(files, args.strict_schema)
        merged = merge_data(dfs)
        merged = process_timestamps(merged)

        if args.sort:
            merged = sort_data(merged)

        save_output(merged, args.output_csv)
        print_summary(merged)

    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
