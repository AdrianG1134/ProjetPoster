from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split

from config import get_default_config
from data import PreparedDataset, prepare_dataset, save_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial CV with GroupKFold by tile for parcel transformer."
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Input CSV path.")
    parser.add_argument("--prepared-npz", type=str, default=None, help="Optional prepared NPZ path.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs_transformer/spatial_cv_groupkfold",
        help="Root output directory.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold folds.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation ratio from train_val in each fold.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--index-filter", type=str, default=None, help="Comma-separated indices.")
    parser.add_argument("--min-obs", type=int, default=None)
    parser.add_argument("--time-grid-frequency", type=str, default=None)
    parser.add_argument("--min-px-count", type=int, default=None)
    parser.add_argument("--max-cloud-scene", type=float, default=None)
    parser.add_argument("--label-group-col", type=str, default=None)

    parser.add_argument(
        "--train-extra-args",
        type=str,
        default="--loss-type focal --focal-gamma 1.5 --class-weighting --class-weight-power 0.5 --standardize-features --no-use-group-task",
        help="Raw extra args passed to train.py for each fold.",
    )
    return parser.parse_args()


def newest_run_dir(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.glob("temporal_transformer_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _stratify_target_if_possible(labels: np.ndarray) -> Optional[np.ndarray]:
    bincount = np.bincount(labels)
    if len(bincount) < 2 or bincount.min() < 2:
        return None
    return labels


def build_fold_dataset(
    prepared: PreparedDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> PreparedDataset:
    return PreparedDataset(
        features=prepared.features,
        day_of_year=prepared.day_of_year,
        observed_mask=prepared.observed_mask,
        labels=prepared.labels,
        group_labels=prepared.group_labels,
        parcel_ids=prepared.parcel_ids,
        parcel_tiles=prepared.parcel_tiles,
        feature_names=prepared.feature_names,
        label_names=prepared.label_names,
        group_label_names=prepared.group_label_names,
        time_grid=prepared.time_grid,
        splits={
            "train": np.asarray(train_idx, dtype=np.int64),
            "val": np.asarray(val_idx, dtype=np.int64),
            "test": np.asarray(test_idx, dtype=np.int64),
        },
    )


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not (0.0 <= args.val_size < 1.0):
        raise ValueError("val_size must be in [0, 1).")

    cfg = get_default_config()
    if args.csv_path is not None:
        cfg.data.csv_path = args.csv_path
        cfg.data.prepared_npz_path = None
    if args.prepared_npz is not None:
        cfg.data.prepared_npz_path = args.prepared_npz
    if args.index_filter is not None:
        cfg.data.index_filter = [x.strip() for x in args.index_filter.split(",") if x.strip()]
    if args.min_obs is not None:
        cfg.data.min_obs_per_parcel = args.min_obs
    if args.time_grid_frequency is not None:
        cfg.data.time_grid_frequency = args.time_grid_frequency
    if args.min_px_count is not None:
        cfg.data.min_px_count = args.min_px_count
    if args.max_cloud_scene is not None:
        cfg.data.max_cloud_scene = args.max_cloud_scene
    if args.label_group_col is not None:
        cfg.data.label_group_col = args.label_group_col

    if cfg.data.csv_path is None and cfg.data.prepared_npz_path is None:
        raise ValueError("Provide --csv-path or --prepared-npz.")

    prepared = prepare_dataset(cfg.data)
    n_samples = prepared.labels.shape[0]
    groups = prepared.parcel_tiles.astype(str)
    unique_tiles = np.unique(groups)
    if unique_tiles.shape[0] < args.n_splits:
        raise ValueError(
            f"Not enough unique tiles for GroupKFold: {unique_tiles.shape[0]} < n_splits={args.n_splits}."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=args.n_splits)
    indices = np.arange(n_samples, dtype=np.int64)
    extra_args = shlex.split(args.train_extra_args) if args.train_extra_args.strip() else []

    rows: list[dict[str, object]] = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        gkf.split(indices, y=prepared.labels, groups=groups),
        start=1,
    ):
        fold_dir = output_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if args.val_size > 0 and train_val_idx.shape[0] > 1:
            stratify_tv = _stratify_target_if_possible(prepared.labels[train_val_idx])
            try:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=args.val_size,
                    random_state=args.seed + fold_id,
                    shuffle=True,
                    stratify=stratify_tv,
                )
            except ValueError:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=args.val_size,
                    random_state=args.seed + fold_id,
                    shuffle=True,
                    stratify=None,
                )
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=np.int64)

        fold_prepared = build_fold_dataset(
            prepared=prepared,
            train_idx=np.sort(train_idx),
            val_idx=np.sort(val_idx),
            test_idx=np.sort(test_idx),
        )
        fold_npz = fold_dir / "prepared_fold.npz"
        save_prepared_dataset(fold_prepared, fold_npz)

        cmd = [
            sys.executable,
            "parcel_transformer/train.py",
            "--prepared-npz",
            str(fold_npz),
            "--output-dir",
            str(fold_dir),
            "--seed",
            str(args.seed + fold_id),
            *extra_args,
        ]
        print(f"\n=== FOLD {fold_id}/{args.n_splits} ===")
        print(" ".join(shlex.quote(x) for x in cmd))
        completed = subprocess.run(cmd, check=False)

        run_dir = newest_run_dir(fold_dir)
        test_macro_f1 = None
        test_accuracy = None
        test_weighted_f1 = None
        if run_dir is not None:
            test_metrics = run_dir / "test_metrics.json"
            if test_metrics.exists():
                with test_metrics.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                test_macro_f1 = float(payload.get("f1_macro", 0.0))
                test_accuracy = float(payload.get("accuracy", 0.0))
                test_weighted_f1 = float(payload.get("f1_weighted", 0.0))

        rows.append(
            {
                "fold": fold_id,
                "return_code": int(completed.returncode),
                "run_dir": str(run_dir) if run_dir is not None else "",
                "test_macro_f1": test_macro_f1,
                "test_accuracy": test_accuracy,
                "test_weighted_f1": test_weighted_f1,
                "n_train": int(train_idx.shape[0]),
                "n_val": int(val_idx.shape[0]),
                "n_test": int(test_idx.shape[0]),
            }
        )

    valid_scores = [float(r["test_macro_f1"]) for r in rows if isinstance(r["test_macro_f1"], float)]
    summary = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_samples": int(n_samples),
        "n_tiles": int(unique_tiles.shape[0]),
        "folds": rows,
        "test_macro_f1_mean": float(np.mean(valid_scores)) if valid_scores else None,
        "test_macro_f1_std": float(np.std(valid_scores)) if valid_scores else None,
    }

    summary_json = output_root / "spatial_cv_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    summary_csv = output_root / "spatial_cv_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "return_code",
                "test_macro_f1",
                "test_accuracy",
                "test_weighted_f1",
                "n_train",
                "n_val",
                "n_test",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SPATIAL CV SUMMARY ===")
    for row in rows:
        print(
            f"fold={row['fold']:>2} "
            f"macro_f1={row['test_macro_f1']} "
            f"acc={row['test_accuracy']} "
            f"w_f1={row['test_weighted_f1']}"
        )
    print(f"mean_macro_f1={summary['test_macro_f1_mean']} std={summary['test_macro_f1_std']}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
