from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from config import get_default_config
from data import build_dataloaders, prepare_dataset, standardize_prepared_features
from model import TemporalTransformerClassifier
from utils import resolve_device, save_json, setup_logger


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pooling: str,
    return_attention: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    parcel_ids: list[str] = []

    attention_sum: Optional[np.ndarray] = None
    attention_count: Optional[np.ndarray] = None

    for batch in dataloader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        labels = batch["label"].to(device=device, dtype=torch.long)

        outputs = model(
            features=features,
            day_of_year=day_of_year,
            observed_mask=observed_mask,
            return_attention=return_attention,
        )
        logits = outputs["logits"]

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        parcel_ids.extend(batch["parcel_id"])

        if return_attention and outputs.get("attention_maps"):
            last_layer = outputs["attention_maps"][-1]  # [B, H, S, S]
            attention = _reduce_attention_to_dates(
                attention_maps=last_layer,
                observed_mask=observed_mask,
                pooling=pooling,
            )
            batch_sum = attention.sum(axis=0)
            batch_count = observed_mask.sum(dim=0).cpu().numpy().astype(np.float64)
            if attention_sum is None:
                attention_sum = batch_sum
                attention_count = batch_count
            else:
                attention_sum += batch_sum
                attention_count += batch_count

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0))
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    preds_np = logits_np.argmax(axis=1) if logits_np.size else np.empty((0,), dtype=np.int64)

    output: dict[str, Any] = {
        "logits": logits_np,
        "labels": labels_np,
        "preds": preds_np,
        "parcel_ids": np.asarray(parcel_ids, dtype=object),
    }
    if return_attention and attention_sum is not None and attention_count is not None:
        temporal_importance = attention_sum / np.clip(attention_count, a_min=1e-6, a_max=None)
        output["temporal_attention"] = temporal_importance
    return output


def _reduce_attention_to_dates(
    attention_maps: torch.Tensor,
    observed_mask: torch.Tensor,
    pooling: str,
) -> np.ndarray:
    """
    Convert last-layer attention maps to per-date importance scores.
    """
    attn = attention_maps.mean(dim=1)  # [B, S, S], average over heads
    mask = observed_mask.float()

    if pooling == "cls":
        # CLS token attends to temporal tokens.
        cls_to_dates = attn[:, 0, 1:]
        score = cls_to_dates * mask
        return score.cpu().numpy()

    # Mean pooling: average attention received by each date token.
    token_to_token = attn
    score = token_to_token.mean(dim=1)
    score = score * mask
    return score.cpu().numpy()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        zero_division=0,
    )

    metrics["confusion_matrix"] = cm
    metrics["classification_report_dict"] = report_dict
    metrics["classification_report_text"] = report_text
    return metrics


def analyze_errors_by_class(
    confusion: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    analysis: list[dict[str, Any]] = []
    num_classes = confusion.shape[0]
    for class_idx in range(num_classes):
        support = int(confusion[class_idx, :].sum())
        tp = int(confusion[class_idx, class_idx])
        fn = int(support - tp)

        off_diag = confusion[class_idx, :].copy()
        off_diag[class_idx] = 0
        if off_diag.sum() > 0:
            most_confused_idx = int(np.argmax(off_diag))
            most_confused_label = label_names[most_confused_idx]
            most_confused_count = int(off_diag[most_confused_idx])
        else:
            most_confused_label = None
            most_confused_count = 0

        analysis.append(
            {
                "class_name": label_names[class_idx],
                "support": support,
                "true_positive": tp,
                "false_negative": fn,
                "main_confusion_target": most_confused_label,
                "main_confusion_count": most_confused_count,
            }
        )
    return analysis


def plot_confusion_matrix(
    confusion: np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    normalize: bool = False,
) -> None:
    cm = confusion.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = cm / np.clip(row_sum, a_min=1e-12, a_max=None)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        annot=False,
        cbar=True,
        square=True,
    )
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
    pooling: str,
    return_attention: bool = False,
) -> dict[str, Any]:
    if len(dataloader.dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset split.")

    predictions = predict_model(
        model=model,
        dataloader=dataloader,
        device=device,
        pooling=pooling,
        return_attention=return_attention,
    )
    metrics = compute_classification_metrics(
        y_true=predictions["labels"],
        y_pred=predictions["preds"],
        label_names=label_names,
    )
    metrics["predictions"] = predictions
    metrics["error_analysis"] = analyze_errors_by_class(
        confusion=metrics["confusion_matrix"],
        label_names=label_names,
    )
    return metrics


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "classification_report": metrics["classification_report_dict"],
        "error_analysis": metrics["error_analysis"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Temporal Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output-dir", type=str, default="eval_outputs", help="Output directory for artifacts.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--config-json", type=str, default=None, help="Optional config.json from training run.")
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "evaluate.log")

    cfg = get_default_config()
    if args.config_json is not None:
        with Path(args.config_json).open("r", encoding="utf-8") as f:
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
        raise ValueError("Provide --csv-path or --prepared-npz (or set prepared_npz_path in config).")

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
            "Applied train-only feature standardization (eps=%.1e) before evaluation.",
            cfg.train.standardize_eps,
        )

    device = resolve_device(args.device)
    dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=args.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint.get("metadata", {})
    num_group_classes = int(metadata.get("num_group_classes", 0) or 0)
    model = TemporalTransformerClassifier(
        input_dim=prepared.num_features,
        num_classes=prepared.num_classes,
        cfg=cfg.model,
        num_group_classes=num_group_classes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    metrics = evaluate_split(
        model=model,
        dataloader=dataloaders[args.split],
        device=device,
        label_names=prepared.label_names,
        pooling=cfg.model.pooling,
        return_attention=True,
    )

    save_json(_serialize_metrics(metrics), output_dir / f"{args.split}_metrics.json")
    with (output_dir / f"{args.split}_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=prepared.label_names,
        output_path=output_dir / f"{args.split}_confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=prepared.label_names,
        output_path=output_dir / f"{args.split}_confusion_matrix_normalized.png",
        normalize=True,
    )

    predictions = metrics["predictions"]
    if "temporal_attention" in predictions:
        att_df = pd.DataFrame(
            {
                "date": pd.to_datetime(prepared.time_grid),
                "day_of_year": pd.to_datetime(prepared.time_grid).dayofyear,
                "mean_attention": predictions["temporal_attention"],
            }
        )
        att_df.to_csv(output_dir / f"{args.split}_temporal_attention.csv", index=False)

    logger.info(
        "%s | accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
        args.split.upper(),
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
    )


if __name__ == "__main__":
    main()
