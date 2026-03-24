from __future__ import annotations

import argparse

from config import get_default_config
from data import prepare_dataset
from utils import save_json, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NPZ dataset from long Sentinel-2 CSV.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to long CSV.")
    parser.add_argument("--output-npz", type=str, required=True, help="Path to output NPZ.")
    parser.add_argument("--index-filter", type=str, default=None, help="Comma-separated list, e.g. NDVI,NDMI,EVI.")
    parser.add_argument("--time-grid-frequency", type=str, default=None, help="e.g. 5D")
    parser.add_argument("--min-obs", type=int, default=4)
    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default="parcel")
    parser.add_argument("--min-px-count", type=int, default=0)
    parser.add_argument("--max-cloud-scene", type=float, default=None)
    parser.add_argument(
        "--label-group-col",
        type=str,
        default=None,
        help="Optional group label column name (default from config: label_group).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.data.csv_path = args.csv_path
    cfg.data.save_prepared_npz_path = args.output_npz
    cfg.data.time_grid_frequency = args.time_grid_frequency
    cfg.data.min_obs_per_parcel = args.min_obs
    cfg.data.split_method = args.split_method
    cfg.data.min_px_count = args.min_px_count
    cfg.data.max_cloud_scene = args.max_cloud_scene
    if args.label_group_col:
        cfg.data.label_group_col = args.label_group_col
    if args.index_filter:
        cfg.data.index_filter = [x.strip() for x in args.index_filter.split(",") if x.strip()]

    logger = setup_logger("prepare_dataset.log")
    prepared = prepare_dataset(cfg.data)
    logger.info(
        "Prepared dataset saved: N=%d, T=%d, F=%d, classes=%d",
        prepared.features.shape[0],
        prepared.features.shape[1],
        prepared.features.shape[2],
        prepared.num_classes,
    )

    save_json(
        {
            "num_parcels": int(prepared.features.shape[0]),
            "seq_len": int(prepared.features.shape[1]),
            "num_features": int(prepared.features.shape[2]),
            "num_classes": int(prepared.num_classes),
            "num_group_classes": int(prepared.num_group_classes),
            "feature_names": prepared.feature_names,
            "label_names": prepared.label_names,
            "group_label_names": prepared.group_label_names,
            "split_sizes": {k: int(v.shape[0]) for k, v in prepared.splits.items()},
        },
        "prepared_dataset_summary.json",
    )
    logger.info("Summary saved to prepared_dataset_summary.json")


if __name__ == "__main__":
    main()
