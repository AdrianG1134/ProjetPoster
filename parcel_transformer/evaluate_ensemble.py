from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from config import get_default_config
from data import build_dataloaders, prepare_dataset, standardize_prepared_features
from evaluate import analyze_errors_by_class, compute_classification_metrics, plot_confusion_matrix
from model import TemporalTransformerClassifier
from utils import resolve_device, save_json, setup_logger


def configure_hierarchical_constraint_from_metadata(
    model: TemporalTransformerClassifier,
    metadata: dict[str, Any],
    logger,
) -> None:
    enabled = bool(metadata.get("hierarchical_constraint", False))
    compat = metadata.get("class_group_compat")
    if not enabled or compat is None:
        return

    try:
        compat_tensor = torch.tensor(compat, dtype=torch.float32, device=next(model.parameters()).device)
        model.configure_hierarchical_constraint(
            class_group_compat=compat_tensor,
            weight=float(metadata.get("hierarchical_constraint_weight", 1.0)),
            eps=float(metadata.get("hierarchical_constraint_eps", 1.0e-6)),
            enabled=True,
        )
        logger.info(
            "Applied hierarchical constraint for checkpoint model (weight=%.3f).",
            float(metadata.get("hierarchical_constraint_weight", 1.0)),
        )
    except Exception as exc:
        logger.warning("Could not apply hierarchical constraint from metadata: %s", exc)


def checkpoint_uses_reliability(checkpoint: dict[str, Any]) -> bool:
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict) and "reliability_aware" in metadata:
        return bool(metadata.get("reliability_aware", False))

    state_dict = checkpoint.get("model_state_dict", {})
    if not isinstance(state_dict, dict):
        return False

    for raw_key in state_dict.keys():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        if key.startswith("reliability_proj.") or key == "reliability_gate_logit":
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of Temporal Transformer checkpoints.")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="One or more checkpoint paths (.pt).",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Glob pattern for checkpoints, e.g. outputs_transformer/seeds_*/**/best_model.pt",
    )
    parser.add_argument("--config-json", type=str, default=None, help="Optional config.json override.")
    parser.add_argument("--output-dir", type=str, default="eval_outputs_ensemble", help="Output directory.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--ensemble-weighting",
        type=str,
        choices=["uniform", "val_macro_f1"],
        default="uniform",
        help="How to weight checkpoints in the ensemble.",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=1.0,
        help="Exponent applied to non-uniform weights before normalization.",
    )

    parser.add_argument("--csv-path", type=str, default=None, help="CSV input path (if not using NPZ).")
    parser.add_argument("--prepared-npz", type=str, default=None, help="Prepared NPZ input path.")
    parser.add_argument("--min-obs", type=int, default=None)
    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default=None)
    parser.add_argument("--time-grid-frequency", type=str, default=None)
    parser.add_argument("--index-filter", type=str, default=None)
    parser.add_argument("--min-px-count", type=int, default=None)
    parser.add_argument("--max-cloud-scene", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default=None)
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    raw_paths: list[str] = []
    if args.checkpoints:
        raw_paths.extend(args.checkpoints)
    if args.checkpoint_glob:
        raw_paths.extend(glob.glob(args.checkpoint_glob, recursive=True))

    if not raw_paths:
        raise ValueError("Provide at least one checkpoint via --checkpoints or --checkpoint-glob.")

    deduped: list[Path] = []
    seen: set[str] = set()
    for p in raw_paths:
        resolved = str(Path(p).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(Path(p))

    for p in deduped:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return deduped


def load_config(args: argparse.Namespace, checkpoints: list[Path]) -> tuple[Any, Optional[Path]]:
    cfg = get_default_config()

    config_path: Optional[Path] = None
    if args.config_json is not None:
        config_path = Path(args.config_json)
    else:
        auto_path = checkpoints[0].parent / "config.json"
        if auto_path.exists():
            config_path = auto_path

    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg.data = cfg.data.__class__(**payload.get("data", {}))
        cfg.model = cfg.model.__class__(**payload.get("model", {}))
        cfg.train = cfg.train.__class__(**payload.get("train", {}))

    if args.csv_path is not None:
        cfg.data.csv_path = args.csv_path
        cfg.data.prepared_npz_path = None
    if args.prepared_npz is not None:
        cfg.data.prepared_npz_path = args.prepared_npz
    if args.min_obs is not None:
        cfg.data.min_obs_per_parcel = args.min_obs
    if args.split_method is not None:
        cfg.data.split_method = args.split_method
    if args.time_grid_frequency is not None:
        cfg.data.time_grid_frequency = args.time_grid_frequency
    if args.index_filter is not None:
        cfg.data.index_filter = [x.strip() for x in args.index_filter.split(",") if x.strip()]
    if args.min_px_count is not None:
        cfg.data.min_px_count = args.min_px_count
    if args.max_cloud_scene is not None:
        cfg.data.max_cloud_scene = args.max_cloud_scene

    if args.d_model is not None:
        cfg.model.d_model = args.d_model
    if args.n_heads is not None:
        cfg.model.n_heads = args.n_heads
    if args.n_layers is not None:
        cfg.model.n_layers = args.n_layers
    if args.ff_dim is not None:
        cfg.model.dim_feedforward = args.ff_dim
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    if args.pooling is not None:
        cfg.model.pooling = args.pooling

    if args.csv_path is None and args.prepared_npz is None and cfg.data.prepared_npz_path is None:
        raise ValueError("Provide --csv-path or --prepared-npz (or ensure config includes prepared_npz_path).")

    return cfg, config_path


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def resolve_model_scores(
    checkpoint_paths: list[Path],
    checkpoints: list[dict[str, Any]],
    logger,
) -> list[float]:
    scores: list[float] = []
    for ckpt_path, checkpoint in zip(checkpoint_paths, checkpoints):
        score: Optional[float] = None
        val_metrics_path = ckpt_path.parent / "val_metrics.json"
        if val_metrics_path.exists():
            try:
                with val_metrics_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                score = _safe_float(payload.get("f1_macro"))
            except Exception as exc:
                logger.warning("Could not read %s: %s", val_metrics_path, exc)
        if score is None:
            score = _safe_float(checkpoint.get("best_metric"))
        if score is None:
            logger.warning("No usable validation score found for %s; defaulting to 0.", ckpt_path)
            score = 0.0
        scores.append(max(0.0, score))
    return scores


def build_model_weights(
    checkpoint_paths: list[Path],
    checkpoints: list[dict[str, Any]],
    weighting: str,
    weight_power: float,
    logger,
) -> tuple[np.ndarray, list[float]]:
    n_models = len(checkpoint_paths)
    if n_models == 0:
        raise ValueError("No checkpoints available to build ensemble weights.")

    if weighting == "uniform":
        return np.full((n_models,), 1.0 / float(n_models), dtype=np.float32), [1.0] * n_models

    raw_scores = resolve_model_scores(checkpoint_paths=checkpoint_paths, checkpoints=checkpoints, logger=logger)
    scores = np.asarray(raw_scores, dtype=np.float64)
    scores = np.clip(scores, a_min=0.0, a_max=None)

    if weight_power <= 0:
        raise ValueError("--weight-power must be > 0.")
    if weight_power != 1.0:
        scores = np.power(scores, weight_power)

    total = float(scores.sum())
    if total <= 0:
        logger.warning("All model scores are zero; falling back to uniform ensemble weights.")
        return np.full((n_models,), 1.0 / float(n_models), dtype=np.float32), raw_scores

    weights = (scores / total).astype(np.float32)
    return weights, raw_scores


@torch.no_grad()
def predict_ensemble(
    models: list[torch.nn.Module],
    model_weights: np.ndarray,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if len(models) == 0:
        raise ValueError("No models were loaded for ensemble prediction.")
    if model_weights.shape[0] != len(models):
        raise ValueError("model_weights size must match number of models.")

    for model in models:
        model.eval()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_parcel_ids: list[str] = []

    for batch in dataloader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        quality_features = batch["quality_features"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)

        logits_sum: Optional[torch.Tensor] = None
        for i, model in enumerate(models):
            outputs = model(
                features=features,
                day_of_year=day_of_year,
                observed_mask=observed_mask,
                quality_features=quality_features,
                return_attention=False,
            )
            logits = outputs["logits"] * float(model_weights[i])
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        assert logits_sum is not None
        all_logits.append(logits_sum.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_parcel_ids.extend(batch["parcel_id"])

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=np.float32)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    preds_np = logits_np.argmax(axis=1) if logits_np.size else np.empty((0,), dtype=np.int64)
    return {
        "logits": logits_np,
        "labels": labels_np,
        "preds": preds_np,
        "parcel_ids": np.asarray(all_parcel_ids, dtype=object),
    }


def main() -> None:
    args = parse_args()
    checkpoint_paths = resolve_checkpoint_paths(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "evaluate_ensemble.log")

    cfg, config_path = load_config(args, checkpoint_paths)
    if config_path is not None:
        logger.info("Loaded config from %s", config_path)
    logger.info("Using %d checkpoints in ensemble.", len(checkpoint_paths))
    logger.info("Checkpoints: %s", [str(p) for p in checkpoint_paths])
    save_json(
        {
            "checkpoints": [str(p) for p in checkpoint_paths],
            "split": args.split,
            "device": args.device,
            "batch_size": args.batch_size,
            "ensemble_weighting": args.ensemble_weighting,
            "weight_power": args.weight_power,
            "config": cfg.to_dict(),
        },
        output_dir / "ensemble_config.json",
    )

    prepared = prepare_dataset(cfg.data)
    if args.split not in prepared.splits:
        raise ValueError(f"Split '{args.split}' not found in prepared dataset.")
    if prepared.splits[args.split].shape[0] == 0:
        raise ValueError(f"Split '{args.split}' is empty.")

    if cfg.train.standardize_features:
        standardize_prepared_features(
            prepared=prepared,
            train_indices=prepared.splits["train"],
            eps=cfg.train.standardize_eps,
        )
        logger.info(
            "Applied train-only feature standardization (eps=%.1e) before ensemble evaluation.",
            cfg.train.standardize_eps,
        )

    device = resolve_device(args.device)
    dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=args.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dataloader = dataloaders[args.split]

    models: list[torch.nn.Module] = []
    loaded_checkpoints: list[dict[str, Any]] = []
    for ckpt_path in checkpoint_paths:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)

        metadata = checkpoint.get("metadata", {})
        uses_reliability = checkpoint_uses_reliability(checkpoint)
        num_group_classes = int(metadata.get("num_group_classes", 0) or 0)
        model_cfg = replace(cfg.model, reliability_aware=uses_reliability)
        model = TemporalTransformerClassifier(
            input_dim=prepared.num_features,
            num_classes=prepared.num_classes,
            cfg=model_cfg,
            num_group_classes=num_group_classes,
        ).to(device)
        label_names_meta = metadata.get("label_names")
        if label_names_meta is not None and list(label_names_meta) != prepared.label_names:
            raise ValueError(
                f"Label names mismatch for checkpoint {ckpt_path}. "
                "Check that all ensemble models were trained on the same class set."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        configure_hierarchical_constraint_from_metadata(model=model, metadata=metadata, logger=logger)
        models.append(model)
        loaded_checkpoints.append(checkpoint)

    model_weights, raw_scores = build_model_weights(
        checkpoint_paths=checkpoint_paths,
        checkpoints=loaded_checkpoints,
        weighting=args.ensemble_weighting,
        weight_power=args.weight_power,
        logger=logger,
    )
    for ckpt_path, raw_score, w in zip(checkpoint_paths, raw_scores, model_weights):
        logger.info(
            "Ensemble weight | checkpoint=%s | raw_score=%.6f | weight=%.6f",
            ckpt_path,
            float(raw_score),
            float(w),
        )

    predictions = predict_ensemble(
        models=models,
        model_weights=model_weights,
        dataloader=dataloader,
        device=device,
    )
    metrics = compute_classification_metrics(
        y_true=predictions["labels"],
        y_pred=predictions["preds"],
        label_names=prepared.label_names,
    )
    metrics["error_analysis"] = analyze_errors_by_class(
        confusion=metrics["confusion_matrix"],
        label_names=prepared.label_names,
    )

    metrics_payload = {
        "accuracy": metrics["accuracy"],
        "recall_macro": metrics["recall_macro"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "classification_report": metrics["classification_report_dict"],
        "error_analysis": metrics["error_analysis"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "n_models": len(models),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "ensemble_weighting": args.ensemble_weighting,
        "weight_power": args.weight_power,
        "model_weights": [float(x) for x in model_weights.tolist()],
        "model_raw_scores": [float(x) for x in raw_scores],
    }
    save_json(metrics_payload, output_dir / f"{args.split}_ensemble_metrics.json")
    with (output_dir / f"{args.split}_ensemble_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])

    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=prepared.label_names,
        output_path=output_dir / f"{args.split}_ensemble_confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=prepared.label_names,
        output_path=output_dir / f"{args.split}_ensemble_confusion_matrix_normalized.png",
        normalize=True,
    )

    true_labels = [prepared.label_names[i] for i in predictions["labels"]]
    pred_labels = [prepared.label_names[i] for i in predictions["preds"]]
    pred_df = pd.DataFrame(
        {
            "parcel_id": predictions["parcel_ids"],
            "y_true_idx": predictions["labels"],
            "y_pred_idx": predictions["preds"],
            "y_true_label": true_labels,
            "y_pred_label": pred_labels,
        }
    )
    pred_df.to_csv(output_dir / f"{args.split}_ensemble_predictions.csv", index=False)

    logger.info(
        "ENSEMBLE %s | n_models=%d | accuracy=%.4f recall_macro=%.4f macro_f1=%.4f weighted_f1=%.4f",
        args.split.upper(),
        len(models),
        metrics["accuracy"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
    )
    logger.info("Artifacts saved in %s", output_dir)


if __name__ == "__main__":
    main()
