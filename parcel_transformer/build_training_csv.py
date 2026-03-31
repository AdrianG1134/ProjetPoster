from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build long training CSV for parcel_transformer from indices CSV + labels CSV."
    )
    parser.add_argument(
        "--indices-csv",
        type=str,
        default="data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d.csv",
        help="Long indices CSV (must include date, tile, ID_PARCEL, index, value_mean, px_count, cloud_scene).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="export.csv",
        help="Parcel labels CSV.",
    )
    parser.add_argument("--id-col", type=str, default="ID_PARCEL", help="Parcel ID column in both files.")
    parser.add_argument(
        "--label-col",
        type=str,
        default="CODE_CULTU",
        help="Label column to use from labels CSV.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="Optional group label column (e.g. CODE_GROUP). If provided, outputs 'label_group'.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/s2_herault_2024_full_year_5day_cloudmask_fast/indices_parcelles_2024-01-01_2024-12-31_win5d_with_labels.csv",
        help="Output long CSV ready for parcel_transformer.",
    )
    parser.add_argument(
        "--indices-filter",
        type=str,
        default=None,
        help="Optional comma-separated indices to keep (e.g., NDVI,NDMI,NDWI,EVI).",
    )
    parser.add_argument(
        "--max-cloud-scene",
        type=float,
        default=None,
        help="Optional max cloud_scene filter.",
    )
    parser.add_argument(
        "--min-px-count",
        type=int,
        default=None,
        help="Optional min px_count filter.",
    )
    parser.add_argument(
        "--min-parcels-per-class",
        type=int,
        default=0,
        help="Drop rare classes with fewer parcels than this threshold (0 = keep all).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size when reading indices CSV.",
    )
    return parser.parse_args()


def normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0+$", "", regex=True)
    return out


def _mode_or_empty(series: pd.Series) -> str:
    series = series.fillna("").astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return ""
    mode = series.mode()
    if mode.empty:
        return ""
    return str(sorted(mode.astype(str).tolist())[0])


def consolidate_labels(
    labels_df: pd.DataFrame,
    id_col: str,
    label_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    labels_df = labels_df.dropna(subset=[id_col, label_col]).copy()
    labels_df[id_col] = normalize_id(labels_df[id_col])
    labels_df[label_col] = labels_df[label_col].astype(str).str.strip()
    labels_df = labels_df[labels_df[label_col] != ""].copy()
    if group_col is not None and group_col in labels_df.columns:
        labels_df[group_col] = labels_df[group_col].fillna("").astype(str).str.strip()

    if labels_df.empty:
        raise ValueError("No valid labels after cleanup.")

    # Resolve duplicated parcel IDs by taking mode label (deterministic tie-break via sorted mode).
    grouped = labels_df.groupby(id_col)
    agg_spec = {
        label_col: lambda s: sorted(s.mode().astype(str).tolist())[0]
    }
    if group_col is not None and group_col in labels_df.columns:
        agg_spec[group_col] = _mode_or_empty
    consolidated = grouped.agg(agg_spec).reset_index()
    return consolidated


def main() -> None:
    args = parse_args()

    indices_path = Path(args.indices_csv)
    labels_path = Path(args.labels_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not indices_path.exists():
        raise FileNotFoundError(f"indices CSV not found: {indices_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    label_usecols = [args.id_col, args.label_col]
    if args.group_col:
        label_usecols.append(args.group_col)
    labels_raw = pd.read_csv(labels_path, usecols=label_usecols)
    labels = consolidate_labels(
        labels_raw,
        id_col=args.id_col,
        label_col=args.label_col,
        group_col=args.group_col,
    )

    if args.min_parcels_per_class > 0:
        class_counts = labels[args.label_col].value_counts()
        valid_classes = set(class_counts[class_counts >= args.min_parcels_per_class].index.astype(str))
        labels = labels[labels[args.label_col].isin(valid_classes)].copy()
        if labels.empty:
            raise ValueError(
                "All labels removed by min_parcels_per_class. Lower this threshold."
            )

    label_map = dict(zip(labels[args.id_col], labels[args.label_col]))
    group_map = (
        dict(zip(labels[args.id_col], labels[args.group_col]))
        if args.group_col and args.group_col in labels.columns
        else None
    )
    keep_indices: Optional[set[str]] = None
    if args.indices_filter:
        keep_indices = {x.strip() for x in args.indices_filter.split(",") if x.strip()}
        if not keep_indices:
            keep_indices = None

    required_input_cols = ["date", "tile", args.id_col, "index", "value_mean", "px_count", "cloud_scene"]
    expected_output_cols = [
        "ID_PARCEL",
        "date",
        "index",
        "value_mean",
        "px_count",
        "cloud_scene",
        "tile",
        "label",
    ]
    if group_map is not None:
        expected_output_cols.append("label_group")

    rows_read = 0
    rows_written = 0
    parcels_written: set[str] = set()
    header_written = False

    for chunk in tqdm(
        pd.read_csv(indices_path, usecols=required_input_cols, chunksize=args.chunksize),
        desc="Building training CSV",
        unit="chunk",
    ):
        rows_read += len(chunk)
        chunk = chunk.dropna(subset=["date", "tile", args.id_col, "index", "value_mean"]).copy()
        if chunk.empty:
            continue

        chunk[args.id_col] = normalize_id(chunk[args.id_col])
        chunk["index"] = chunk["index"].astype(str).str.strip()

        if keep_indices is not None:
            chunk = chunk[chunk["index"].isin(keep_indices)].copy()
            if chunk.empty:
                continue

        if args.max_cloud_scene is not None:
            chunk["cloud_scene"] = pd.to_numeric(chunk["cloud_scene"], errors="coerce")
            chunk = chunk[chunk["cloud_scene"] <= args.max_cloud_scene].copy()
            if chunk.empty:
                continue

        if args.min_px_count is not None:
            chunk["px_count"] = pd.to_numeric(chunk["px_count"], errors="coerce")
            chunk = chunk[chunk["px_count"] >= args.min_px_count].copy()
            if chunk.empty:
                continue

        chunk["label"] = chunk[args.id_col].map(label_map)
        chunk = chunk.dropna(subset=["label"]).copy()
        if chunk.empty:
            continue

        if group_map is not None:
            chunk["label_group"] = (
                chunk[args.id_col].map(group_map).fillna("").astype(str).str.strip()
            )

        chunk["ID_PARCEL"] = chunk[args.id_col]
        chunk["tile"] = chunk["tile"].astype(str).str.strip()
        chunk["date"] = chunk["date"].astype(str).str.strip()
        chunk["label"] = chunk["label"].astype(str).str.strip()

        out_chunk = chunk[expected_output_cols]
        out_chunk.to_csv(output_path, mode="a", index=False, header=not header_written)
        header_written = True

        rows_written += len(out_chunk)
        parcels_written.update(out_chunk["ID_PARCEL"].unique().tolist())

    if not header_written:
        raise RuntimeError(
            "No rows written. Check ID alignment, label column, and optional filters."
        )

    summary = {
        "indices_csv": str(indices_path),
        "labels_csv": str(labels_path),
        "output_csv": str(output_path),
        "id_col": args.id_col,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "indices_filter": sorted(list(keep_indices)) if keep_indices is not None else None,
        "max_cloud_scene": args.max_cloud_scene,
        "min_px_count": args.min_px_count,
        "min_parcels_per_class": args.min_parcels_per_class,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "unique_parcels_with_labels": len(parcels_written),
        "unique_labels_kept": int(labels[args.label_col].nunique()),
        "unique_groups_kept": int(labels[args.group_col].nunique()) if args.group_col and args.group_col in labels.columns else 0,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[OK] CSV generated: {output_path}")
    print(f"[OK] Summary saved: {summary_path}")
    print(f"Rows written: {rows_written:,} | Parcels: {len(parcels_written):,}")


if __name__ == "__main__":
    main()
