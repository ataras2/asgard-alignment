#!/usr/bin/env python3
"""
compare_toml.py

Compare two TOML files with nested tables.
Classify base items (non-table values) as shared or not-shared.
Summarize values (arrays -> shape, strings -> value, scalars -> value, etc.).
Output two pandas tables and optionally export them.

Usage:
    python compare_toml.py path/to/a.toml path/to/b.toml
    # Optional exports:
    python compare_toml.py a.toml b.toml --export-csv shared.csv notshared.csv
    python compare_toml.py a.toml b.toml --export-xlsx report.xlsx
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

# TOML loader: tomllib (py>=3.11) then tomli fallback
try:
    import tomllib  # type: ignore
    def load_toml_bytes(data: bytes) -> Dict[str, Any]:
        return tomllib.loads(data.decode("utf-8"))
except Exception:
    import tomli  # type: ignore
    def load_toml_bytes(data: bytes) -> Dict[str, Any]:
        return tomli.loads(data.decode("utf-8"))

import numpy as np
import pandas as pd
from datetime import datetime, date, time


def load_toml_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return load_toml_bytes(f.read())


def is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def flatten_base_items(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested TOML dict to dotted keys. Only keep base items (non-dict values).
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if is_mapping(v):
            out.update(flatten_base_items(v, key))
        else:
            out[key] = v
    return out


def classify_type(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        return "number" if isinstance(value, (int, float, np.integer, np.floating)) else "bool"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    return type(value).__name__


def summarize_value(value: Any, maxlen: int = 120) -> str:
    t = classify_type(value)
    if t == "array":
        try:
            arr = np.array(value, dtype=object)
            return f"shape={arr.shape}"
        except Exception as e:
            return f"array (shape=unknown; err={e})"
    elif t == "string":
        s = value
        s = s if len(s) <= maxlen else s[:maxlen - 3] + "..."
        return s
    elif t in ("number", "bool"):
        return str(value)
    elif t in ("datetime", "date", "time"):
        try:
            return value.isoformat()
        except Exception:
            return repr(value)
    else:
        r = repr(value)
        return r if len(r) <= maxlen else r[:maxlen - 3] + "..."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare two TOML files and summarize base items.")
    parser.add_argument("file_a", type=Path, help="First TOML file")
    parser.add_argument("file_b", type=Path, help="Second TOML file")
    parser.add_argument("--export-csv", nargs=2, metavar=("SHARED_CSV", "NOTSHARED_CSV"),
                        help="Export shared and not-shared tables to CSV files")
    parser.add_argument("--export-xlsx", metavar="XLSX_PATH",
                        help="Export both tables to a single Excel workbook (two sheets)")
    args = parser.parse_args(argv)

    if not args.file_a.exists() or not args.file_b.exists():
        print("Error: one or both input files do not exist.", file=sys.stderr)
        return 2

    data_a = load_toml_file(args.file_a)
    data_b = load_toml_file(args.file_b)

    flat_a = flatten_base_items(data_a)
    flat_b = flatten_base_items(data_b)

    keys_a = set(flat_a.keys())
    keys_b = set(flat_b.keys())

    shared_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    # Build Shared table
    shared_rows = []
    for k in shared_keys:
        va = flat_a[k]
        vb = flat_b[k]
        shared_rows.append({
            "key": k,
            "type_a": classify_type(va),
            "summary_a": summarize_value(va),
            "type_b": classify_type(vb),
            "summary_b": summarize_value(vb),
        })
    df_shared = pd.DataFrame(shared_rows, columns=["key", "type_a", "summary_a", "type_b", "summary_b"])

    # Build Not-shared table
    notshared_rows = []
    for k in only_a:
        v = flat_a[k]
        notshared_rows.append({
            "key": k,
            "present_in": args.file_a.name,
            "type": classify_type(v),
            "summary": summarize_value(v),
        })
    for k in only_b:
        v = flat_b[k]
        notshared_rows.append({
            "key": k,
            "present_in": args.file_b.name,
            "type": classify_type(v),
            "summary": summarize_value(v),
        })
    df_notshared = pd.DataFrame(notshared_rows, columns=["key", "present_in", "type", "summary"])

    # Pretty print small previews
    pd.set_option("display.max_colwidth", 120)
    print(f"\n=== Shared base items ({len(df_shared)}) ===")
    print(df_shared.to_string(index=False))

    print(f"\n=== Not-shared base items ({len(df_notshared)}) ===")
    print(df_notshared.to_string(index=False))

    # Optional exports
    if args.export_csv:
        shared_csv, notshared_csv = args.export_csv
        df_shared.to_csv(shared_csv, index=False)
        df_notshared.to_csv(notshared_csv, index=False)
        print(f"\nSaved CSVs -> {shared_csv}, {notshared_csv}")

    if args.export_xlsx:
        xlsx_path = args.export_xlsx
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
            df_shared.to_excel(xl, index=False, sheet_name="shared")
            df_notshared.to_excel(xl, index=False, sheet_name="not_shared")
        print(f"Saved Excel workbook -> {xlsx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
