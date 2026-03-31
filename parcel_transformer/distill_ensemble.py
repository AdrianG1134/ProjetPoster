from __future__ import annotations

import argparse
import glob
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

from config import ExperimentConfig, get_default_config
from data import build_dataloaders, prepare_dataset, standardize_prepared_features
from evaluate import evaluate_split, plot_confusion_matrix
from model import TemporalTransformerClassifier
from utils import (
    EarlyStopping,
    compute_class_weights,
    create_run_dir,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill an ensemble of Temporal Transformer checkpoints into a single student model."
    )

    parser.add_argument(
        "--teacher-checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="One or more teacher checkpoint paths (.pt).",
    )
    parser.add_argument(
        "--teacher-checkpoint-glob",
        type=str,
        default=None,
        help="Glob pattern for teacher checkpoints, e.g. outputs_transformer/seeds_*/**/best_model.pt",
    )
    parser.add_argument("--config-json", type=str, default=None, help="Optional config.json override.")

    parser.add_argument("--output-dir", type=str, default="outputs_transformer/distillation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--csv-path", type=str, default=None, help="CSV input path (if not using NPZ).")
    parser.add_argument("--prepared-npz", type=str, default=None, help="Prepared NPZ input path.")
    parser.add_argument("--min-obs", type=int, default=None)
    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default=None)
    parser.add_argument("--time-grid-frequency", type=str, default=None)
    parser.add_argument("--index-filter", type=str, default=None)
    parser.add_argument("--min-px-count", type=int, default=None)
    parser.add_argument("--max-cloud-scene", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)

    parser.add_argument(
        "--standardize-features",
        dest="standardize_features",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-standardize-features",
        dest="standardize_features",
        action="store_false",
        default=None,
    )
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--no-class-weighting", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=None)

    parser.add_argument(
        "--hard-label-weight",
        type=float,
        default=0.4,
        help="Weight for hard-label cross-entropy. Distillation weight is (1 - hard-label-weight).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature.",
    )

    parser.add_argument("--student-d-model", type=int, default=None)
    parser.add_argument("--student-n-heads", type=int, default=None)
    parser.add_argument("--student-n-layers", type=int, default=None)
    parser.add_argument("--student-ff-dim", type=int, default=None)
    parser.add_argument("--student-dropout", type=float, default=None)
    parser.add_argument("--student-pooling", type=str, choices=["cls", "mean"], default=None)

    return parser.parse_args()


def resolve_teacher_checkpoints(args: argparse.Namespace) -> list[Path]:
    raw_paths: list[str] = []
    if args.teacher_checkpoints:
        raw_paths.extend(args.teacher_checkpoints)
    if args.teacher_checkpoint_glob:
        raw_paths.extend(glob.glob(args.teacher_checkpoint_glob, recursive=True))

    if not raw_paths:
        raise ValueError("Provide teacher checkpoints via --teacher-checkpoints and/or --teacher-checkpoint-glob.")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in raw_paths:
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(Path(path))

    for path in deduped:
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    return sorted(deduped)


def load_config_from_json(path: Path, cfg: ExperimentConfig) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg.data = cfg.data.__class__(**payload.get("data", {}))
    cfg.model = cfg.model.__class__(**payload.get("model", {}))
    cfg.train = cfg.train.__class__(**payload.get("train", {}))
    return cfg


def load_base_config(args: argparse.Namespace, teacher_checkpoints: list[Path]) -> tuple[ExperimentConfig, Optional[Path]]:
    cfg = get_default_config()

    config_path: Optional[Path] = None
    if args.config_json is not None:
        config_path = Path(args.config_json)
    else:
        auto_path = teacher_checkpoints[0].parent / "config.json"
        if auto_path.exists():
            config_path = auto_path

    if config_path is not None:
        cfg = load_config_from_json(config_path, cfg)

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

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.scheduler_patience is not None:
        cfg.train.scheduler_patience = args.scheduler_patience
    if args.scheduler_factor is not None:
        cfg.train.scheduler_factor = args.scheduler_factor
    if args.min_learning_rate is not None:
        cfg.train.min_learning_rate = args.min_learning_rate
    if args.early_stopping_patience is not None:
        cfg.train.early_stopping_patience = args.early_stopping_patience
    if args.gradient_clip_norm is not None:
        cfg.train.gradient_clip_norm = args.gradient_clip_norm
    if args.standardize_features is not None:
        cfg.train.standardize_features = args.standardize_features
    if args.class_weight_power is not None:
        cfg.train.class_weight_power = args.class_weight_power
    if args.seed is not None:
        cfg.train.seed = args.seed

    if args.class_weighting:
        cfg.train.class_weighting = True
    if args.no_class_weighting:
        cfg.train.class_weighting = False

    if args.csv_path is None and args.prepared_npz is None and cfg.data.prepared_npz_path is None:
        raise ValueError("Provide --csv-path or --prepared-npz, or ensure config includes prepared_npz_path.")

    return cfg, config_path


def build_student_model_cfg(cfg: ExperimentConfig, args: argparse.Namespace):
    student_cfg = replace(cfg.model)
    if args.student_d_model is not None:
        student_cfg.d_model = args.student_d_model
    if args.student_n_heads is not None:
        student_cfg.n_heads = args.student_n_heads
    if args.student_n_layers is not None:
        student_cfg.n_layers = args.student_n_layers
    if args.student_ff_dim is not None:
        student_cfg.dim_feedforward = args.student_ff_dim
    if args.student_dropout is not None:
        student_cfg.dropout = args.student_dropout
    if args.student_pooling is not None:
        student_cfg.pooling = args.student_pooling
    return student_cfg


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ExperimentConfig):
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.train.scheduler_factor,
            patience=cfg.train.scheduler_patience,
            min_lr=cfg.train.min_learning_rate,
        )
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.min_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")


def load_teacher_model_cfg(teacher_checkpoint: Path, fallback_cfg: ExperimentConfig):
    cfg_path = teacher_checkpoint.parent / "config.json"
    if not cfg_path.exists():
        return replace(fallback_cfg.model)

    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return fallback_cfg.model.__class__(**payload.get("model", {}))


def load_teachers(
    teacher_checkpoints: list[Path],
    cfg: ExperimentConfig,
    prepared,
    device: torch.device,
) -> list[nn.Module]:
    teachers: list[nn.Module] = []
    for checkpoint_path in teacher_checkpoints:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        metadata = checkpoint.get("metadata", {})
        label_names_meta = metadata.get("label_names")
        if label_names_meta is not None and list(label_names_meta) != prepared.label_names:
            raise ValueError(
                f"Label names mismatch for teacher checkpoint {checkpoint_path}. "
                "Teachers must share the exact same class set."
            )

        teacher_model_cfg = load_teacher_model_cfg(checkpoint_path, cfg)
        teacher_num_group_classes = int(metadata.get("num_group_classes", 0) or 0)

        teacher = TemporalTransformerClassifier(
            input_dim=prepared.num_features,
            num_classes=prepared.num_classes,
            cfg=teacher_model_cfg,
            num_group_classes=teacher_num_group_classes,
        ).to(device)
        teacher.load_state_dict(checkpoint["model_state_dict"])
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        teachers.append(teacher)

    return teachers


def run_distill_train_epoch(
    student: nn.Module,
    teachers: list[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    hard_criterion: nn.Module,
    hard_label_weight: float,
    temperature: float,
    gradient_clip_norm: Optional[float],
) -> dict[str, float]:
    student.train()
    for teacher in teachers:
        teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for batch in dataloader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        labels = batch["label"].to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits_sum: Optional[torch.Tensor] = None
            for teacher in teachers:
                outputs_t = teacher(
                    features=features,
                    day_of_year=day_of_year,
                    observed_mask=observed_mask,
                    return_attention=False,
                )
                logits_t = outputs_t["logits"]
                teacher_logits_sum = logits_t if teacher_logits_sum is None else (teacher_logits_sum + logits_t)
            assert teacher_logits_sum is not None
            teacher_logits = teacher_logits_sum / float(len(teachers))

        outputs_s = student(
            features=features,
            day_of_year=day_of_year,
            observed_mask=observed_mask,
            return_attention=False,
        )
        student_logits = outputs_s["logits"]

        ce_loss = hard_criterion(student_logits, labels)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss = (hard_label_weight * ce_loss) + ((1.0 - hard_label_weight) * kd_loss)

        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_kd += float(kd_loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(student_logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.empty((0,), dtype=np.int64)

    return {
        "loss": total_loss / max(total_samples, 1),
        "ce_loss": total_ce / max(total_samples, 1),
        "kd_loss": total_kd / max(total_samples, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)) if total_samples > 0 else float("nan"),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_samples > 0 else float("nan"),
    }


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for batch in dataloader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        labels = batch["label"].to(device=device, dtype=torch.long)

        outputs = model(
            features=features,
            day_of_year=day_of_year,
            observed_mask=observed_mask,
            return_attention=False,
        )
        logits = outputs["logits"]
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(labels.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.empty((0,), dtype=np.int64)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)) if total_samples > 0 else float("nan"),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_samples > 0 else float("nan"),
    }


def save_eval_artifacts(
    split_name: str,
    metrics: dict,
    label_names: list[str],
    run_dir: Path,
    time_grid: np.ndarray,
) -> None:
    serializable_metrics = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "classification_report": metrics["classification_report_dict"],
        "error_analysis": metrics["error_analysis"],
    }
    save_json(serializable_metrics, run_dir / f"{split_name}_metrics.json")

    with (run_dir / f"{split_name}_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])

    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=run_dir / f"{split_name}_confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=run_dir / f"{split_name}_confusion_matrix_normalized.png",
        normalize=True,
    )

    predictions = metrics.get("predictions", {})
    if "temporal_attention" in predictions:
        attention_df = pd.DataFrame(
            {
                "date": pd.to_datetime(time_grid),
                "day_of_year": pd.to_datetime(time_grid).dayofyear,
                "mean_attention": predictions["temporal_attention"],
            }
        )
        attention_df.to_csv(run_dir / f"{split_name}_temporal_attention.csv", index=False)


def main() -> None:
    args = parse_args()
    teacher_checkpoints = resolve_teacher_checkpoints(args)
    cfg, config_path = load_base_config(args, teacher_checkpoints)
    student_model_cfg = build_student_model_cfg(cfg, args)

    if cfg.train.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if cfg.train.learning_rate <= 0:
        raise ValueError("lr must be > 0.")
    if cfg.train.batch_size <= 0:
        raise ValueError("batch-size must be > 0.")
    if not 0.0 <= args.hard_label_weight <= 1.0:
        raise ValueError("hard-label-weight must be in [0, 1].")
    if args.temperature <= 0:
        raise ValueError("temperature must be > 0.")

    set_seed(cfg.train.seed)
    run_dir = create_run_dir(cfg.data.output_dir if args.output_dir is None else args.output_dir, prefix="distill_student")
    logger = setup_logger(run_dir / "distill.log")
    logger.info("Run directory: %s", run_dir)
    if config_path is not None:
        logger.info("Loaded base config from %s", config_path)
    logger.info("Using %d teacher checkpoints.", len(teacher_checkpoints))
    logger.info("Teacher checkpoints: %s", [str(p) for p in teacher_checkpoints])

    save_json(
        {
            "teacher_checkpoints": [str(p) for p in teacher_checkpoints],
            "base_config_path": str(config_path) if config_path is not None else None,
            "experiment_config": cfg.to_dict(),
            "student_model_config": {
                "d_model": student_model_cfg.d_model,
                "n_heads": student_model_cfg.n_heads,
                "n_layers": student_model_cfg.n_layers,
                "dim_feedforward": student_model_cfg.dim_feedforward,
                "dropout": student_model_cfg.dropout,
                "pooling": student_model_cfg.pooling,
            },
            "distillation": {
                "hard_label_weight": args.hard_label_weight,
                "temperature": args.temperature,
            },
        },
        run_dir / "distill_config.json",
    )

    prepared = prepare_dataset(cfg.data)
    logger.info(
        "Prepared dataset: N=%d, T=%d, F=%d, classes=%d",
        prepared.features.shape[0],
        prepared.seq_len,
        prepared.num_features,
        prepared.num_classes,
    )
    logger.info("Features: %s", ", ".join(prepared.feature_names))
    logger.info("Classes: %s", ", ".join(prepared.label_names))
    split_sizes = {k: int(v.shape[0]) for k, v in prepared.splits.items()}
    logger.info("Split sizes: %s", split_sizes)

    if split_sizes.get("train", 0) == 0:
        raise ValueError("Train split is empty.")
    if split_sizes.get("val", 0) == 0:
        logger.warning("Validation split is empty. Early stopping disabled.")
    if split_sizes.get("test", 0) == 0:
        raise ValueError("Test split is empty.")

    if cfg.train.standardize_features:
        scaler_stats = standardize_prepared_features(
            prepared=prepared,
            train_indices=prepared.splits["train"],
            eps=cfg.train.standardize_eps,
        )
        save_json(
            {
                "feature_names": prepared.feature_names,
                "mean": scaler_stats["mean"].tolist(),
                "std": scaler_stats["std"].tolist(),
                "count_per_feature": scaler_stats["count_per_feature"].tolist(),
            },
            run_dir / "feature_scaler.json",
        )
        logger.info("Applied train-only feature standardization (eps=%.1e).", cfg.train.standardize_eps)
    else:
        logger.info("Feature standardization disabled.")

    device = resolve_device(args.device if args.device is not None else cfg.train.device)
    logger.info("Device: %s", device)

    dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        train_sampler=None,
        train_augmentation=None,
    )

    teachers = load_teachers(
        teacher_checkpoints=teacher_checkpoints,
        cfg=cfg,
        prepared=prepared,
        device=device,
    )

    student = TemporalTransformerClassifier(
        input_dim=prepared.num_features,
        num_classes=prepared.num_classes,
        cfg=student_model_cfg,
        num_group_classes=0,
    ).to(device)

    class_weights_t: Optional[torch.Tensor] = None
    if cfg.train.class_weighting:
        train_labels = prepared.labels[prepared.splits["train"]]
        class_weights_np = compute_class_weights(
            train_labels,
            num_classes=prepared.num_classes,
            power=cfg.train.class_weight_power,
        )
        class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
        logger.info(
            "Class weighting enabled | power=%.3f | min=%.3f max=%.3f",
            cfg.train.class_weight_power,
            float(class_weights_t.min().item()),
            float(class_weights_t.max().item()),
        )
    else:
        logger.info("Class weighting disabled.")

    hard_criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)
    early_stopping = (
        EarlyStopping(patience=cfg.train.early_stopping_patience, mode="max")
        if split_sizes.get("val", 0) > 0
        else None
    )

    best_val_f1 = float("-inf")
    best_model_path = run_dir / "best_model.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_distill_train_epoch(
            student=student,
            teachers=teachers,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            hard_criterion=hard_criterion,
            hard_label_weight=args.hard_label_weight,
            temperature=args.temperature,
            gradient_clip_norm=cfg.train.gradient_clip_norm,
        )

        if split_sizes.get("val", 0) > 0:
            val_metrics = run_eval_epoch(
                model=student,
                dataloader=dataloaders["val"],
                device=device,
                criterion=hard_criterion,
            )
            val_f1 = val_metrics["f1_macro"]
        else:
            val_metrics = {"loss": float("nan"), "accuracy": float("nan"), "f1_macro": float("nan")}
            val_f1 = train_metrics["f1_macro"]

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %03d | lr=%.6f | train_loss=%.4f (ce=%.4f kd=%.4f) train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch,
            current_lr,
            train_metrics["loss"],
            train_metrics["ce_loss"],
            train_metrics["kd_loss"],
            train_metrics["f1_macro"],
            val_metrics["loss"],
            val_metrics["f1_macro"],
        )
        history.append(
            {
                "epoch": float(epoch),
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_kd_loss": train_metrics["kd_loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                path=best_model_path,
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_f1,
                metadata={
                    "label_names": prepared.label_names,
                    "feature_names": prepared.feature_names,
                    "pooling": student_model_cfg.pooling,
                    "distillation_temperature": args.temperature,
                    "distillation_hard_label_weight": args.hard_label_weight,
                    "teacher_checkpoints": [str(p) for p in teacher_checkpoints],
                    "student_model_config": {
                        "d_model": student_model_cfg.d_model,
                        "n_heads": student_model_cfg.n_heads,
                        "n_layers": student_model_cfg.n_layers,
                        "dim_feedforward": student_model_cfg.dim_feedforward,
                        "dropout": student_model_cfg.dropout,
                        "pooling": student_model_cfg.pooling,
                    },
                },
            )
            logger.info("Saved new best student model (val_f1=%.4f).", best_val_f1)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if early_stopping is not None and early_stopping.step(val_f1):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    logger.info("Training finished. Loading best checkpoint from %s", best_model_path)
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])

    if split_sizes.get("val", 0) > 0:
        val_eval = evaluate_split(
            model=student,
            dataloader=dataloaders["val"],
            device=device,
            label_names=prepared.label_names,
            pooling=student_model_cfg.pooling,
            return_attention=True,
        )
        save_eval_artifacts("val", val_eval, prepared.label_names, run_dir, prepared.time_grid)
        logger.info(
            "VAL | acc=%.4f macro_f1=%.4f weighted_f1=%.4f",
            val_eval["accuracy"],
            val_eval["f1_macro"],
            val_eval["f1_weighted"],
        )

    test_eval = evaluate_split(
        model=student,
        dataloader=dataloaders["test"],
        device=device,
        label_names=prepared.label_names,
        pooling=student_model_cfg.pooling,
        return_attention=True,
    )
    save_eval_artifacts("test", test_eval, prepared.label_names, run_dir, prepared.time_grid)
    logger.info(
        "TEST | acc=%.4f macro_f1=%.4f weighted_f1=%.4f",
        test_eval["accuracy"],
        test_eval["f1_macro"],
        test_eval["f1_weighted"],
    )
    logger.info("Artifacts saved in %s", run_dir)


if __name__ == "__main__":
    main()
