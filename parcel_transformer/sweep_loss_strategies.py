from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini sweep for loss strategies (focal, balanced_softmax, logit_adjusted)."
    )
    parser.add_argument("--csv-path", type=str, required=True, help="Training CSV path.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs_transformer/sweep_loss_strategies",
        help="Root folder for sweep outputs.",
    )
    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default="tile")
    parser.add_argument("--index-filter", type=str, default="NDVI,NDMI,NDWI,EVI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument(
        "--logit-taus",
        type=str,
        default="0.5,1.5",
        help="Comma-separated tau values for logit_adjusted.",
    )
    parser.add_argument(
        "--standardize-features",
        dest="standardize_features",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-standardize-features",
        dest="standardize_features",
        action="store_false",
    )
    parser.add_argument(
        "--class-weighting",
        dest="class_weighting",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-class-weighting",
        dest="class_weighting",
        action="store_false",
    )
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra raw args passed to train.py (quoted string).",
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


def run_variant(
    variant_name: str,
    variant_args: list[str],
    common_args: list[str],
    output_root: Path,
) -> dict[str, object]:
    variant_output = output_root / variant_name
    variant_output.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "parcel_transformer/train.py", *common_args, "--output-dir", str(variant_output), *variant_args]
    print(f"\n=== RUN {variant_name} ===")
    print(" ".join(shlex.quote(x) for x in cmd))

    started_at = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(cmd, check=False)
    ended_at = datetime.now().isoformat(timespec="seconds")

    run_dir = newest_run_dir(variant_output)
    test_metrics = None
    if run_dir is not None:
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                test_metrics = json.load(f)

    result: dict[str, object] = {
        "variant": variant_name,
        "return_code": int(completed.returncode),
        "started_at": started_at,
        "ended_at": ended_at,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "test_accuracy": None,
        "test_macro_f1": None,
        "test_weighted_f1": None,
    }
    if test_metrics is not None:
        result["test_accuracy"] = float(test_metrics.get("accuracy", 0.0))
        result["test_macro_f1"] = float(test_metrics.get("f1_macro", 0.0))
        result["test_weighted_f1"] = float(test_metrics.get("f1_weighted", 0.0))
    return result


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--csv-path",
        args.csv_path,
        "--split-method",
        args.split_method,
        "--index-filter",
        args.index_filter,
        "--seed",
        str(args.seed),
        "--class-weight-power",
        str(args.class_weight_power),
        "--no-use-group-task",
    ]
    common_args.append("--standardize-features" if args.standardize_features else "--no-standardize-features")
    common_args.append("--class-weighting" if args.class_weighting else "--no-class-weighting")
    if args.extra_args.strip():
        common_args.extend(shlex.split(args.extra_args))

    taus = [x.strip() for x in args.logit_taus.split(",") if x.strip()]
    variants: list[tuple[str, list[str]]] = [
        (
            "focal",
            ["--loss-type", "focal", "--focal-gamma", str(args.focal_gamma)],
        ),
        (
            "balanced_softmax",
            ["--loss-type", "balanced_softmax"],
        ),
    ]
    for tau_str in taus:
        float(tau_str)  # validate
        safe_tau = tau_str.replace(".", "p")
        variants.append(
            (
                f"logit_adjusted_tau_{safe_tau}",
                ["--loss-type", "logit_adjusted", "--logit-adjust-tau", tau_str],
            )
        )

    results: list[dict[str, object]] = []
    for variant_name, variant_args in variants:
        results.append(
            run_variant(
                variant_name=variant_name,
                variant_args=variant_args,
                common_args=common_args,
                output_root=output_root,
            )
        )

    def score_key(row: dict[str, object]) -> float:
        v = row.get("test_macro_f1")
        return float(v) if isinstance(v, (int, float)) else -1.0

    ranked = sorted(results, key=score_key, reverse=True)

    summary_json = output_root / "sweep_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=True)

    summary_csv = output_root / "sweep_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "return_code",
                "test_macro_f1",
                "test_accuracy",
                "test_weighted_f1",
                "run_dir",
                "started_at",
                "ended_at",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked)

    print("\n=== SWEEP SUMMARY (sorted by test_macro_f1) ===")
    for row in ranked:
        print(
            f"{row['variant']:<28} "
            f"macro_f1={row['test_macro_f1']} "
            f"acc={row['test_accuracy']} "
            f"w_f1={row['test_weighted_f1']}"
        )
    print(f"\nSaved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
