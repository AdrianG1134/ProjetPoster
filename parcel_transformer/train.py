from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from config import ExperimentConfig, get_default_config
from data import (
    TemporalAugmentationConfig,
    build_dataloaders,
    prepare_dataset,
    standardize_prepared_features,
)
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
    parser = argparse.ArgumentParser(description="Temporal Transformer for parcel crop classification")

    parser.add_argument("--csv-path", type=str, default=None, help="Path to long CSV input.")
    parser.add_argument("--prepared-npz", type=str, default=None, help="Load prepared dataset from NPZ.")
    parser.add_argument("--save-prepared-npz", type=str, default=None, help="Optional output NPZ path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory.")
    parser.add_argument(
        "--label-group-col",
        type=str,
        default=None,
        help="Optional group label column name in CSV (default from config: label_group).",
    )

    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default=None)
    parser.add_argument("--time-grid-frequency", type=str, default=None, help="e.g. 5D. Default uses observed dates.")
    parser.add_argument("--index-filter", type=str, default=None, help="Comma-separated list of indices.")
    parser.add_argument("--min-obs", type=int, default=None, help="Min observed dates per parcel.")
    parser.add_argument("--max-cloud-scene", type=float, default=None)
    parser.add_argument("--min-px-count", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default=None)
    parser.add_argument(
        "--reliability-aware",
        dest="reliability_aware",
        action="store_true",
        default=None,
        help="Inject reliability signals (observed mask / cloud / px_count) into temporal encoding and pooling.",
    )
    parser.add_argument(
        "--no-reliability-aware",
        dest="reliability_aware",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--pretrained-encoder-checkpoint",
        type=str,
        default=None,
        help="Optional SSL checkpoint (.pt) used to initialize backbone encoder weights before supervised training.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--standardize-features",
        dest="standardize_features",
        action="store_true",
        default=None,
        help="Apply train-only feature standardization using observed mask.",
    )
    parser.add_argument(
        "--no-standardize-features",
        dest="standardize_features",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["cross_entropy", "focal", "balanced_softmax", "logit_adjusted"],
        default=None,
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=None,
        help="Tau for logit-adjusted cross entropy (used when --loss-type logit_adjusted).",
    )
    parser.add_argument(
        "--use-group-task",
        dest="use_group_task",
        action="store_true",
        default=None,
        help="Enable auxiliary group classification head (requires label_group column).",
    )
    parser.add_argument(
        "--no-use-group-task",
        dest="use_group_task",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--group-loss-weight",
        type=float,
        default=None,
        help="Weight for auxiliary group loss when --use-group-task is enabled.",
    )
    parser.add_argument(
        "--hierarchical-constraint",
        dest="hierarchical_constraint",
        action="store_true",
        default=None,
        help="Enable hierarchical constrained decoding (group probabilities constrain class logits).",
    )
    parser.add_argument(
        "--no-hierarchical-constraint",
        dest="hierarchical_constraint",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--hierarchical-constraint-weight",
        type=float,
        default=None,
        help="Lambda weight for hierarchical log-compatibility added to class logits.",
    )
    parser.add_argument(
        "--hierarchical-constraint-eps",
        type=float,
        default=None,
        help="Numerical epsilon for hierarchical log-compatibility term.",
    )
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default=None)
    parser.add_argument("--class-weighting", action="store_true")
    parser.add_argument("--no-class-weighting", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=None)
    parser.add_argument(
        "--weighted-sampler",
        dest="weighted_sampler",
        action="store_true",
        default=None,
        help="Use WeightedRandomSampler for the train DataLoader.",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        dest="weighted_sampler",
        action="store_false",
        default=None,
    )
    parser.add_argument("--sampler-power", type=float, default=None)
    parser.add_argument(
        "--temporal-augmentation",
        dest="temporal_augmentation",
        action="store_true",
        default=None,
        help="Enable light temporal augmentation on train batches (time masking + jitter).",
    )
    parser.add_argument(
        "--no-temporal-augmentation",
        dest="temporal_augmentation",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--time-mask-ratio",
        type=float,
        default=None,
        help="Fraction of observed timesteps masked in train augmentation (e.g. 0.05).",
    )
    parser.add_argument(
        "--jitter-std",
        type=float,
        default=None,
        help="Gaussian noise std applied on observed timesteps in train augmentation.",
    )
    parser.add_argument(
        "--phase2-rare-finetune",
        dest="phase2_rare_finetune",
        action="store_true",
        default=None,
        help="Enable 2nd fine-tuning phase focused on rare classes.",
    )
    parser.add_argument(
        "--no-phase2-rare-finetune",
        dest="phase2_rare_finetune",
        action="store_false",
        default=None,
    )
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--phase2-lr", type=float, default=None)
    parser.add_argument("--phase2-sampler-power", type=float, default=None)
    parser.add_argument("--phase2-rare-quantile", type=float, default=None)
    parser.add_argument("--phase2-rare-count-threshold", type=int, default=None)
    parser.add_argument("--phase2-rare-boost", type=float, default=None)
    parser.add_argument("--phase2-early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace, cfg: ExperimentConfig) -> ExperimentConfig:
    if args.csv_path is not None:
        cfg.data.csv_path = args.csv_path
    if args.prepared_npz is not None:
        cfg.data.prepared_npz_path = args.prepared_npz
    if args.save_prepared_npz is not None:
        cfg.data.save_prepared_npz_path = args.save_prepared_npz
    if args.output_dir is not None:
        cfg.data.output_dir = args.output_dir
    if args.label_group_col is not None:
        cfg.data.label_group_col = args.label_group_col

    if args.split_method is not None:
        cfg.data.split_method = args.split_method
    if args.time_grid_frequency is not None:
        cfg.data.time_grid_frequency = args.time_grid_frequency
    if args.index_filter is not None:
        cfg.data.index_filter = [x.strip() for x in args.index_filter.split(",") if x.strip()]
    if args.min_obs is not None:
        cfg.data.min_obs_per_parcel = args.min_obs
    if args.max_cloud_scene is not None:
        cfg.data.max_cloud_scene = args.max_cloud_scene
    if args.min_px_count is not None:
        cfg.data.min_px_count = args.min_px_count

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
    if args.reliability_aware is not None:
        cfg.model.reliability_aware = args.reliability_aware

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.standardize_features is not None:
        cfg.train.standardize_features = args.standardize_features
    if args.loss_type is not None:
        cfg.train.loss_type = args.loss_type
    if args.focal_gamma is not None:
        cfg.train.focal_gamma = args.focal_gamma
    if args.logit_adjust_tau is not None:
        cfg.train.logit_adjust_tau = args.logit_adjust_tau
    if args.use_group_task is not None:
        cfg.train.use_group_task = args.use_group_task
    if args.group_loss_weight is not None:
        cfg.train.group_loss_weight = args.group_loss_weight
    if args.hierarchical_constraint is not None:
        cfg.train.hierarchical_constraint = args.hierarchical_constraint
    if args.hierarchical_constraint_weight is not None:
        cfg.train.hierarchical_constraint_weight = args.hierarchical_constraint_weight
    if args.hierarchical_constraint_eps is not None:
        cfg.train.hierarchical_constraint_eps = args.hierarchical_constraint_eps
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.class_weight_power is not None:
        cfg.train.class_weight_power = args.class_weight_power
    if args.weighted_sampler is not None:
        cfg.train.weighted_sampler = args.weighted_sampler
    if args.sampler_power is not None:
        cfg.train.sampler_power = args.sampler_power
    if args.temporal_augmentation is not None:
        cfg.train.temporal_augmentation = args.temporal_augmentation
    if args.time_mask_ratio is not None:
        cfg.train.time_mask_ratio = args.time_mask_ratio
    if args.jitter_std is not None:
        cfg.train.jitter_std = args.jitter_std
    if args.phase2_rare_finetune is not None:
        cfg.train.phase2_rare_finetune = args.phase2_rare_finetune
    if args.phase2_epochs is not None:
        cfg.train.phase2_epochs = args.phase2_epochs
    if args.phase2_lr is not None:
        cfg.train.phase2_learning_rate = args.phase2_lr
    if args.phase2_sampler_power is not None:
        cfg.train.phase2_sampler_power = args.phase2_sampler_power
    if args.phase2_rare_quantile is not None:
        cfg.train.phase2_rare_quantile = args.phase2_rare_quantile
    if args.phase2_rare_count_threshold is not None:
        cfg.train.phase2_rare_count_threshold = args.phase2_rare_count_threshold
    if args.phase2_rare_boost is not None:
        cfg.train.phase2_rare_boost = args.phase2_rare_boost
    if args.phase2_early_stopping_patience is not None:
        cfg.train.phase2_early_stopping_patience = args.phase2_early_stopping_patience
    if args.early_stopping_patience is not None:
        cfg.train.early_stopping_patience = args.early_stopping_patience
    if args.device is not None:
        cfg.train.device = args.device
    if args.seed is not None:
        cfg.train.seed = args.seed

    if args.class_weighting:
        cfg.train.class_weighting = True
    if args.no_class_weighting:
        cfg.train.class_weighting = False

    return cfg


def _load_pretrained_encoder_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    logger,
) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"pretrained encoder checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        if "encoder_state_dict" in checkpoint and isinstance(checkpoint["encoder_state_dict"], dict):
            source_state = checkpoint["encoder_state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            source_state = checkpoint["model_state_dict"]
        else:
            source_state = {
                k: v
                for k, v in checkpoint.items()
                if isinstance(v, torch.Tensor)
            }
    else:
        raise ValueError("Unsupported checkpoint format for pretrained encoder.")

    allowed_prefixes = (
        "feature_proj.",
        "doy_encoding.",
        "layers.",
        "cls_token",
        "reliability_proj.",
        "reliability_gate_logit",
    )
    normalized_state: dict[str, torch.Tensor] = {}
    for raw_key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        normalized_state[key] = value

    filtered_state = {
        k: v
        for k, v in normalized_state.items()
        if any(k == prefix or k.startswith(prefix) for prefix in allowed_prefixes)
    }
    if len(filtered_state) == 0:
        raise ValueError(
            "No compatible encoder parameters found in pretrained checkpoint. "
            "Expected keys starting with: feature_proj, doy_encoding, layers, cls_token, reliability_* "
            "(optionally prefixed by backbone. and/or module.)."
        )

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    logger.info(
        "Loaded pretrained encoder weights from %s | loaded=%d | missing=%d | unexpected=%d",
        path,
        len(filtered_state),
        len(missing),
        len(unexpected),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
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


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be >= 0.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not torch.any(valid_mask):
                return logits.sum() * 0.0
            logits = logits[valid_mask]
            targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = torch.pow((1.0 - target_probs).clamp(min=0.0), self.gamma)
        loss = -focal_factor * target_log_probs

        if self.class_weights is not None:
            alpha = self.class_weights.gather(0, targets)
            loss = alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: Tensor,
        class_weights: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")
        counts = class_counts.float().clamp(min=1.0)
        self.register_buffer("log_counts", counts.log())
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not torch.any(valid_mask):
                return logits.sum() * 0.0
            logits = logits[valid_mask]
            targets = targets[valid_mask]

        adjusted_logits = logits + self.log_counts.unsqueeze(0)
        loss = F.cross_entropy(adjusted_logits, targets, reduction="none")
        if self.class_weights is not None:
            alpha = self.class_weights.gather(0, targets)
            loss = alpha * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LogitAdjustedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_priors: Tensor,
        tau: float = 1.0,
        class_weights: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if tau < 0:
            raise ValueError("logit_adjust_tau must be >= 0.")
        priors = class_priors.float().clamp(min=1e-12)
        self.register_buffer("logit_adjustment", float(tau) * priors.log())
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not torch.any(valid_mask):
                return logits.sum() * 0.0
            logits = logits[valid_mask]
            targets = targets[valid_mask]

        adjusted_logits = logits + self.logit_adjustment.unsqueeze(0)
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.class_weights,
            reduction=self.reduction,
        )


def build_loss_criterion(
    loss_type: str,
    class_weights: Optional[Tensor],
    class_counts: Tensor,
    class_priors: Tensor,
    focal_gamma: float,
    logit_adjust_tau: float,
    ignore_index: Optional[int] = None,
) -> nn.Module:
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)
    if loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            class_weights=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
    if loss_type == "balanced_softmax":
        return BalancedSoftmaxLoss(
            class_counts=class_counts,
            class_weights=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
    if loss_type == "logit_adjusted":
        return LogitAdjustedCrossEntropyLoss(
            class_priors=class_priors,
            tau=logit_adjust_tau,
            class_weights=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
    raise ValueError(f"Unsupported loss type: {loss_type}")


def build_weighted_train_sampler(
    labels: np.ndarray,
    train_indices: np.ndarray,
    num_classes: int,
    power: float,
) -> tuple[WeightedRandomSampler, dict[str, float]]:
    train_labels = labels[train_indices]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_sample_weights = np.power(1.0 / counts, power)
    class_sample_weights = class_sample_weights / np.clip(class_sample_weights.mean(), a_min=1e-12, a_max=None)
    sample_weights = class_sample_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )
    stats = {
        "min_sample_weight": float(sample_weights.min()),
        "max_sample_weight": float(sample_weights.max()),
        "mean_sample_weight": float(sample_weights.mean()),
    }
    return sampler, stats


def select_rare_classes(
    class_counts: np.ndarray,
    rare_quantile: float,
    rare_count_threshold: Optional[int],
) -> tuple[np.ndarray, float]:
    nonzero_counts = class_counts[class_counts > 0]
    if nonzero_counts.size == 0:
        return np.array([], dtype=np.int64), 0.0

    if rare_count_threshold is not None:
        threshold = float(rare_count_threshold)
    else:
        quantile = float(np.clip(rare_quantile, 0.0, 1.0))
        threshold = float(np.quantile(nonzero_counts, quantile))

    rare_ids = np.where(class_counts <= threshold)[0].astype(np.int64)
    if rare_ids.size == 0:
        rare_ids = np.array([int(np.argmin(class_counts))], dtype=np.int64)
    return rare_ids, threshold


def build_rare_finetune_sampler(
    labels: np.ndarray,
    train_indices: np.ndarray,
    num_classes: int,
    rare_class_ids: np.ndarray,
    power: float,
    rare_boost: float,
) -> tuple[WeightedRandomSampler, dict[str, float]]:
    train_labels = labels[train_indices]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    class_sample_weights = np.ones((num_classes,), dtype=np.float64)
    if rare_class_ids.size > 0:
        rare_scores = np.power(1.0 / counts[rare_class_ids], power)
        rare_scores = rare_scores / np.clip(rare_scores.mean(), a_min=1e-12, a_max=None)
        class_sample_weights[rare_class_ids] = float(rare_boost) * rare_scores

    sample_weights = class_sample_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )
    stats = {
        "min_sample_weight": float(sample_weights.min()),
        "max_sample_weight": float(sample_weights.max()),
        "mean_sample_weight": float(sample_weights.mean()),
    }
    return sampler, stats


def build_class_group_compatibility(
    labels: np.ndarray,
    group_labels: np.ndarray,
    train_indices: np.ndarray,
    num_classes: int,
    num_group_classes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if num_group_classes <= 0:
        raise ValueError("num_group_classes must be > 0.")

    train_labels = labels[train_indices]
    train_groups = group_labels[train_indices]
    valid_mask = train_groups >= 0
    valid_labels = train_labels[valid_mask]
    valid_groups = train_groups[valid_mask]

    counts = np.zeros((num_group_classes, num_classes), dtype=np.float64)
    np.add.at(counts, (valid_groups, valid_labels), 1.0)

    col_sum = counts.sum(axis=0, keepdims=True)
    missing_classes_mask = (col_sum <= 0).reshape(-1)
    if np.any(missing_classes_mask):
        counts[:, missing_classes_mask] = 1.0 / float(num_group_classes)
        col_sum = counts.sum(axis=0, keepdims=True)

    compat = counts / np.clip(col_sum, a_min=1.0e-12, a_max=None)
    links_per_class = (compat > 0).sum(axis=0)
    stats = {
        "valid_samples": float(valid_mask.sum()),
        "missing_classes": float(missing_classes_mask.sum()),
        "mean_links_per_class": float(links_per_class.mean()),
        "max_links_per_class": float(links_per_class.max()),
    }
    return compat.astype(np.float32), stats


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    group_criterion: Optional[nn.Module] = None,
    group_loss_weight: float = 0.0,
    gradient_clip_norm: Optional[float] = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_group_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    total_samples = 0

    progress = tqdm(dataloader, leave=False, desc="train" if is_train else "eval")
    for batch in progress:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        quality_features = batch["quality_features"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        group_labels = batch["group_label"].to(device=device, dtype=torch.long)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                features=features,
                day_of_year=day_of_year,
                observed_mask=observed_mask,
                quality_features=quality_features,
                return_attention=False,
            )
            logits = outputs["logits"]
            main_loss = criterion(logits, labels)
            group_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
            if group_criterion is not None and "group_logits" in outputs:
                group_loss = group_criterion(outputs["group_logits"], group_labels)
            loss = main_loss + group_loss_weight * group_loss

            if is_train:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_group_loss += float(group_loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.empty((0,), dtype=np.int64)

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_group_loss = total_group_loss / max(total_samples, 1)
    epoch_acc = float(accuracy_score(y_true, y_pred)) if total_samples > 0 else float("nan")
    epoch_recall_macro = (
        float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if total_samples > 0
        else float("nan")
    )
    epoch_f1_weighted = (
        float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if total_samples > 0
        else float("nan")
    )
    epoch_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if total_samples > 0 else float("nan")
    return {
        "loss": epoch_loss,
        "group_loss": epoch_group_loss,
        "accuracy": epoch_acc,
        "recall_macro": epoch_recall_macro,
        "f1_weighted": epoch_f1_weighted,
        "f1_macro": epoch_f1,
    }


def save_eval_artifacts(
    split_name: str,
    metrics: dict,
    label_names: list[str],
    run_dir: Path,
    time_grid: np.ndarray,
) -> None:
    serializable_metrics = {
        "exactitude": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "rappel_macro": metrics["recall_macro"],
        "rappel_pondere": metrics["recall_weighted"],
        "f1_macro": metrics["f1_macro"],
        "f1_pondere": metrics["f1_weighted"],
        "rapport_classification": metrics["classification_report_dict"],
        "analyse_erreurs": metrics["error_analysis"],
        "matrice_confusion": metrics["confusion_matrix"].tolist(),
        # Compatibilite descendante
        "accuracy": metrics["accuracy"],
        "recall_macro": metrics["recall_macro"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "classification_report": metrics["classification_report_dict"],
        "error_analysis": metrics["error_analysis"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    save_json(serializable_metrics, run_dir / f"{split_name}_metriques.json")
    save_json(serializable_metrics, run_dir / f"{split_name}_metrics.json")

    with (run_dir / f"{split_name}_rapport_classification.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])
    with (run_dir / f"{split_name}_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])

    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=run_dir / f"{split_name}_matrice_confusion.png",
        normalize=False,
    )
    plot_confusion_matrix(
        confusion=metrics["confusion_matrix"],
        label_names=label_names,
        output_path=run_dir / f"{split_name}_matrice_confusion_normalisee.png",
        normalize=True,
    )
    # Compatibilite descendante
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
        attention_df.to_csv(run_dir / f"{split_name}_attention_temporelle.csv", index=False)
        attention_df.to_csv(run_dir / f"{split_name}_temporal_attention.csv", index=False)


def save_split_comparison_artifacts(split_metrics: dict[str, dict], run_dir: Path) -> None:
    if not split_metrics:
        return

    split_order = list(split_metrics.keys())
    split_labels = {
        "train": "entrainement",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }

    rows: list[dict[str, float | str]] = []
    for split in split_order:
        metrics = split_metrics[split]
        rows.append(
            {
                "jeu": split_labels.get(split, split),
                "exactitude": float(metrics.get("accuracy", float("nan"))),
                "precision_macro": float(metrics.get("precision_macro", float("nan"))),
                "rappel_macro": float(metrics.get("recall_macro", float("nan"))),
                "f1_pondere": float(metrics.get("f1_weighted", float("nan"))),
                "f1_macro": float(metrics.get("f1_macro", float("nan"))),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(run_dir / "resume_metriques_splits.csv", index=False)
    summary_df.to_csv(run_dir / "split_metrics_summary.csv", index=False)

def save_epoch_metric_curves(
    history: list[dict[str, float]],
    phase2_history: list[dict[str, float]],
    run_dir: Path,
) -> None:
    if not history and not phase2_history:
        return

    all_rows: list[dict[str, float]] = []
    for row in history:
        all_rows.append(dict(row))

    phase1_len = len(history)
    for row in phase2_history:
        merged = dict(row)
        merged["epoch"] = float(phase1_len + int(row.get("epoch", 0)))
        all_rows.append(merged)

    history_df = pd.DataFrame(all_rows).sort_values("epoch").reset_index(drop=True)
    history_df.to_csv(run_dir / "history_all_phases.csv", index=False)

    metric_specs = [
        ("accuracy", "Accuracy", "train_acc", "val_acc"),
        ("recall_macro", "Recall (macro)", "train_recall_macro", "val_recall_macro"),
        ("f1_weighted", "F1 (weighted)", "train_f1_weighted", "val_f1_weighted"),
        ("f1_macro", "F1 (macro)", "train_f1_macro", "val_f1_macro"),
    ]

    for metric_key, metric_title, train_col, val_col in metric_specs:
        if train_col not in history_df.columns or val_col not in history_df.columns:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.plot(
            history_df["epoch"],
            history_df[train_col],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="train",
        )
        plt.plot(
            history_df["epoch"],
            history_df[val_col],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="val",
        )

        if phase2_history:
            plt.axvline(
                x=phase1_len + 0.5,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label="phase2 start",
            )

        plt.ylim(0.0, 1.0)
        plt.xlabel("Epoch")
        plt.ylabel(metric_title)
        plt.title(f"{metric_title}: train vs val")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"metric_compare_{metric_key}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    cfg = apply_args_to_config(args, get_default_config())

    if not 0.0 <= cfg.train.time_mask_ratio < 1.0:
        raise ValueError("time_mask_ratio must be in [0, 1).")
    if cfg.train.jitter_std < 0:
        raise ValueError("jitter_std must be >= 0.")
    if cfg.train.phase2_epochs <= 0:
        raise ValueError("phase2_epochs must be > 0.")
    if cfg.train.phase2_learning_rate <= 0:
        raise ValueError("phase2_learning_rate must be > 0.")
    if cfg.train.phase2_sampler_power < 0:
        raise ValueError("phase2_sampler_power must be >= 0.")
    if not 0.0 <= cfg.train.phase2_rare_quantile <= 1.0:
        raise ValueError("phase2_rare_quantile must be in [0, 1].")
    if cfg.train.phase2_rare_boost <= 0:
        raise ValueError("phase2_rare_boost must be > 0.")
    if cfg.train.phase2_early_stopping_patience <= 0:
        raise ValueError("phase2_early_stopping_patience must be > 0.")
    if cfg.train.hierarchical_constraint_weight < 0:
        raise ValueError("hierarchical_constraint_weight must be >= 0.")
    if cfg.train.hierarchical_constraint_eps <= 0:
        raise ValueError("hierarchical_constraint_eps must be > 0.")

    set_seed(cfg.train.seed)
    run_dir = create_run_dir(cfg.data.output_dir, prefix="temporal_transformer")
    logger = setup_logger(run_dir / "train.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration loaded.")
    save_json(cfg.to_dict(), run_dir / "config.json")

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
    if prepared.has_group_labels:
        logger.info("Group classes (%s): %s", cfg.data.label_group_col, ", ".join(prepared.group_label_names))
    else:
        logger.info("No usable group labels detected in column '%s'.", cfg.data.label_group_col)
    split_sizes = {k: int(v.shape[0]) for k, v in prepared.splits.items()}
    logger.info("Split sizes: %s", split_sizes)

    if split_sizes.get("train", 0) == 0:
        raise ValueError("Le split d'entrainement est vide.")
    if split_sizes.get("val", 0) == 0:
        logger.warning("Le split de validation est vide. L'early stopping sera desactive.")
    if split_sizes.get("test", 0) == 0:
        raise ValueError("Le split de test est vide.")

    use_group_task = cfg.train.use_group_task or cfg.train.hierarchical_constraint
    if cfg.train.hierarchical_constraint and not cfg.train.use_group_task:
        logger.info(
            "Hierarchical constrained decoding enabled: group head is activated automatically."
        )
    if use_group_task and not prepared.has_group_labels:
        raise ValueError(
            "Group task enabled but no group labels are available. "
            f"Ensure column '{cfg.data.label_group_col}' exists and is populated."
        )
    if use_group_task and cfg.train.group_loss_weight <= 0:
        raise ValueError("group_loss_weight must be > 0 when group task is enabled.")

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
        logger.info(
            "Applied train-only feature standardization (eps=%.1e).",
            cfg.train.standardize_eps,
        )
    else:
        logger.info("Feature standardization disabled.")

    device = resolve_device(cfg.train.device)
    logger.info("Device: %s", device)

    train_augmentation: Optional[TemporalAugmentationConfig] = None
    if cfg.train.temporal_augmentation:
        train_augmentation = TemporalAugmentationConfig(
            time_mask_ratio=cfg.train.time_mask_ratio,
            jitter_std=cfg.train.jitter_std,
        )
        logger.info(
            "Temporal augmentation enabled | time_mask_ratio=%.3f | jitter_std=%.4f",
            cfg.train.time_mask_ratio,
            cfg.train.jitter_std,
        )
    else:
        logger.info("Temporal augmentation disabled.")

    train_sampler: Optional[WeightedRandomSampler] = None
    if cfg.train.weighted_sampler:
        train_sampler, sampler_stats = build_weighted_train_sampler(
            labels=prepared.labels,
            train_indices=prepared.splits["train"],
            num_classes=prepared.num_classes,
            power=cfg.train.sampler_power,
        )
        logger.info(
            "Weighted sampler enabled | power=%.3f | min=%.3f max=%.3f",
            cfg.train.sampler_power,
            sampler_stats["min_sample_weight"],
            sampler_stats["max_sample_weight"],
        )
    else:
        logger.info("Weighted sampler disabled.")

    if cfg.train.weighted_sampler and cfg.train.class_weighting:
        logger.warning(
            "Both weighted sampler and class weighting are enabled. "
            "This can over-correct minority classes; monitor precision/recall tradeoff."
        )
    if cfg.train.phase2_rare_finetune and cfg.train.weighted_sampler:
        logger.warning(
            "phase2_rare_finetune is enabled, but weighted sampler is also active in phase 1. "
            "For a stricter two-stage setup, use --no-weighted-sampler in phase 1."
        )

    dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        train_sampler=train_sampler,
        train_augmentation=train_augmentation,
    )

    model = TemporalTransformerClassifier(
        input_dim=prepared.num_features,
        num_classes=prepared.num_classes,
        cfg=cfg.model,
        num_group_classes=prepared.num_group_classes if use_group_task else 0,
    ).to(device)
    logger.info("Reliability-aware transformer: %s", "enabled" if cfg.model.reliability_aware else "disabled")
    if args.pretrained_encoder_checkpoint:
        _load_pretrained_encoder_weights(
            model=model,
            checkpoint_path=args.pretrained_encoder_checkpoint,
            device=device,
            logger=logger,
        )

    train_labels = prepared.labels[prepared.splits["train"]]
    class_counts_np = np.bincount(train_labels, minlength=prepared.num_classes).astype(np.float64)
    class_counts_np = np.clip(class_counts_np, a_min=1.0, a_max=None)
    class_priors_np = class_counts_np / class_counts_np.sum()
    class_counts_t = torch.tensor(class_counts_np, dtype=torch.float32, device=device)
    class_priors_t = torch.tensor(class_priors_np, dtype=torch.float32, device=device)

    class_weights: Optional[torch.Tensor] = None
    if cfg.train.class_weighting:
        class_weights_np = compute_class_weights(
            train_labels,
            num_classes=prepared.num_classes,
            power=cfg.train.class_weight_power,
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
        logger.info(
            "Class weighting enabled | power=%.3f | min=%.3f max=%.3f",
            cfg.train.class_weight_power,
            float(class_weights.min().item()),
            float(class_weights.max().item()),
        )
    else:
        logger.info("Class weighting disabled.")

    if cfg.train.loss_type == "balanced_softmax" and cfg.train.class_weighting:
        logger.warning(
            "balanced_softmax with class weighting can over-correct tail classes; "
            "consider disabling class weighting for this loss."
        )

    group_class_weights: Optional[torch.Tensor] = None
    group_criterion: Optional[nn.Module] = None
    group_counts_t: Optional[torch.Tensor] = None
    group_priors_t: Optional[torch.Tensor] = None
    class_group_compat_np: Optional[np.ndarray] = None
    if use_group_task:
        assert prepared.group_labels is not None
        train_group_labels = prepared.group_labels[prepared.splits["train"]]
        valid_group_train = train_group_labels[train_group_labels >= 0]
        if valid_group_train.size == 0:
            raise ValueError("Group task enabled but no valid group labels found in train split.")

        group_counts_np = np.bincount(valid_group_train, minlength=prepared.num_group_classes).astype(np.float64)
        group_counts_np = np.clip(group_counts_np, a_min=1.0, a_max=None)
        group_priors_np = group_counts_np / group_counts_np.sum()
        group_counts_t = torch.tensor(group_counts_np, dtype=torch.float32, device=device)
        group_priors_t = torch.tensor(group_priors_np, dtype=torch.float32, device=device)

        if cfg.train.class_weighting:
            group_class_weights_np = compute_class_weights(
                valid_group_train,
                num_classes=prepared.num_group_classes,
                power=cfg.train.class_weight_power,
            )
            group_class_weights = torch.tensor(group_class_weights_np, dtype=torch.float32, device=device)
            logger.info(
                "Group class weighting enabled | power=%.3f | min=%.3f max=%.3f",
                cfg.train.class_weight_power,
                float(group_class_weights.min().item()),
                float(group_class_weights.max().item()),
            )

        logger.info(
            "Group task enabled | groups=%d | group_loss_weight=%.3f",
            prepared.num_group_classes,
            cfg.train.group_loss_weight,
        )

        if cfg.train.hierarchical_constraint:
            class_group_compat_np, compat_stats = build_class_group_compatibility(
                labels=prepared.labels,
                group_labels=prepared.group_labels,
                train_indices=prepared.splits["train"],
                num_classes=prepared.num_classes,
                num_group_classes=prepared.num_group_classes,
            )
            model.configure_hierarchical_constraint(
                class_group_compat=torch.tensor(class_group_compat_np, dtype=torch.float32, device=device),
                weight=cfg.train.hierarchical_constraint_weight,
                eps=cfg.train.hierarchical_constraint_eps,
                enabled=True,
            )
            save_json(
                {
                    "group_label_names": prepared.group_label_names,
                    "label_names": prepared.label_names,
                    "class_group_compat": class_group_compat_np.tolist(),
                    "weight": cfg.train.hierarchical_constraint_weight,
                    "eps": cfg.train.hierarchical_constraint_eps,
                    "stats": compat_stats,
                },
                run_dir / "hierarchical_constraint.json",
            )
            logger.info(
                "Hierarchical constrained decoding enabled | weight=%.3f | eps=%.1e | missing_classes=%.0f | mean_links=%.2f",
                cfg.train.hierarchical_constraint_weight,
                cfg.train.hierarchical_constraint_eps,
                compat_stats["missing_classes"],
                compat_stats["mean_links_per_class"],
            )

    criterion = build_loss_criterion(
        loss_type=cfg.train.loss_type,
        class_weights=class_weights,
        class_counts=class_counts_t,
        class_priors=class_priors_t,
        focal_gamma=cfg.train.focal_gamma,
        logit_adjust_tau=cfg.train.logit_adjust_tau,
        ignore_index=None,
    )
    if use_group_task:
        assert group_counts_t is not None and group_priors_t is not None
        group_criterion = build_loss_criterion(
            loss_type=cfg.train.loss_type,
            class_weights=group_class_weights,
            class_counts=group_counts_t,
            class_priors=group_priors_t,
            focal_gamma=cfg.train.focal_gamma,
            logit_adjust_tau=cfg.train.logit_adjust_tau,
            ignore_index=-100,
        )

    if cfg.train.loss_type == "cross_entropy":
        logger.info("Loss: CrossEntropy")
    elif cfg.train.loss_type == "focal":
        logger.info("Loss: Focal (gamma=%.3f)", cfg.train.focal_gamma)
    elif cfg.train.loss_type == "balanced_softmax":
        logger.info("Loss: BalancedSoftmax")
    elif cfg.train.loss_type == "logit_adjusted":
        logger.info("Loss: LogitAdjustedCrossEntropy (tau=%.3f)", cfg.train.logit_adjust_tau)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    phase2_history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            group_criterion=group_criterion,
            group_loss_weight=cfg.train.group_loss_weight if use_group_task else 0.0,
            gradient_clip_norm=cfg.train.gradient_clip_norm,
        )

        if split_sizes.get("val", 0) > 0:
            val_metrics = run_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                optimizer=None,
                group_criterion=group_criterion,
                group_loss_weight=cfg.train.group_loss_weight if use_group_task else 0.0,
                gradient_clip_norm=None,
            )
            val_f1 = val_metrics["f1_macro"]
        else:
            val_metrics = {
                "loss": float("nan"),
                "group_loss": float("nan"),
                "accuracy": float("nan"),
                "recall_macro": float("nan"),
                "f1_weighted": float("nan"),
                "f1_macro": float("nan"),
            }
            val_f1 = train_metrics["f1_macro"]

        current_lr = float(optimizer.param_groups[0]["lr"])
        if use_group_task:
            logger.info(
                "Epoch %03d | lr=%.6f | train_loss=%.4f (grp=%.4f) train_f1=%.4f | "
                "val_loss=%.4f (grp=%.4f) val_f1=%.4f",
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["group_loss"],
                train_metrics["f1_macro"],
                val_metrics["loss"],
                val_metrics["group_loss"],
                val_metrics["f1_macro"],
            )
        else:
            logger.info(
                "Epoch %03d | lr=%.6f | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["f1_macro"],
                val_metrics["loss"],
                val_metrics["f1_macro"],
            )
        history.append(
            {
                "epoch": float(epoch),
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_group_loss": train_metrics["group_loss"],
                "train_acc": train_metrics["accuracy"],
                "train_recall_macro": train_metrics["recall_macro"],
                "train_f1_weighted": train_metrics["f1_weighted"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_metrics["loss"],
                "val_group_loss": val_metrics["group_loss"],
                "val_acc": val_metrics["accuracy"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_f1,
                metadata={
                    "label_names": prepared.label_names,
                    "group_label_names": prepared.group_label_names if use_group_task else [],
                    "feature_names": prepared.feature_names,
                    "pooling": cfg.model.pooling,
                    "reliability_aware": bool(cfg.model.reliability_aware),
                    "loss_type": cfg.train.loss_type,
                    "focal_gamma": cfg.train.focal_gamma,
                    "logit_adjust_tau": cfg.train.logit_adjust_tau,
                    "class_weight_power": cfg.train.class_weight_power,
                    "num_group_classes": prepared.num_group_classes if use_group_task else 0,
                    "use_group_task": bool(use_group_task),
                    "hierarchical_constraint": bool(cfg.train.hierarchical_constraint),
                    "hierarchical_constraint_weight": cfg.train.hierarchical_constraint_weight,
                    "hierarchical_constraint_eps": cfg.train.hierarchical_constraint_eps,
                    "class_group_compat": class_group_compat_np.tolist() if class_group_compat_np is not None else None,
                },
            )
            logger.info("Saved new best model (val_f1=%.4f).", best_val_f1)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        if early_stopping is not None and early_stopping.step(val_f1):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    if cfg.train.phase2_rare_finetune:
        logger.info("Starting phase 2 rare-class fine-tuning.")
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        rare_class_ids, rare_threshold = select_rare_classes(
            class_counts=class_counts_np,
            rare_quantile=cfg.train.phase2_rare_quantile,
            rare_count_threshold=cfg.train.phase2_rare_count_threshold,
        )
        rare_class_names = [prepared.label_names[int(i)] for i in rare_class_ids.tolist()]
        logger.info(
            "Phase 2 rare classes selected | threshold=%.1f | n_rare=%d | classes=%s",
            rare_threshold,
            len(rare_class_ids),
            ", ".join(rare_class_names),
        )

        phase2_sampler, phase2_stats = build_rare_finetune_sampler(
            labels=prepared.labels,
            train_indices=prepared.splits["train"],
            num_classes=prepared.num_classes,
            rare_class_ids=rare_class_ids,
            power=cfg.train.phase2_sampler_power,
            rare_boost=cfg.train.phase2_rare_boost,
        )
        logger.info(
            "Phase 2 sampler enabled | power=%.3f | boost=%.3f | min=%.3f max=%.3f",
            cfg.train.phase2_sampler_power,
            cfg.train.phase2_rare_boost,
            phase2_stats["min_sample_weight"],
            phase2_stats["max_sample_weight"],
        )

        phase2_dataloaders = build_dataloaders(
            prepared=prepared,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
            train_sampler=phase2_sampler,
            train_augmentation=train_augmentation,
        )

        phase2_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.phase2_learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        phase2_scheduler = None
        phase2_early_stopping = (
            EarlyStopping(patience=cfg.train.phase2_early_stopping_patience, mode="max")
            if split_sizes.get("val", 0) > 0
            else None
        )

        phase2_best_model_path = run_dir / "best_model_phase2.pt"
        phase2_best_val_f1 = best_val_f1
        for epoch in range(1, cfg.train.phase2_epochs + 1):
            train_metrics = run_epoch(
                model=model,
                dataloader=phase2_dataloaders["train"],
                criterion=criterion,
                device=device,
                optimizer=phase2_optimizer,
                group_criterion=group_criterion,
                group_loss_weight=cfg.train.group_loss_weight if use_group_task else 0.0,
                gradient_clip_norm=cfg.train.gradient_clip_norm,
            )

            if split_sizes.get("val", 0) > 0:
                val_metrics = run_epoch(
                    model=model,
                    dataloader=phase2_dataloaders["val"],
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    group_criterion=group_criterion,
                    group_loss_weight=cfg.train.group_loss_weight if use_group_task else 0.0,
                    gradient_clip_norm=None,
                )
                val_f1 = val_metrics["f1_macro"]
            else:
                val_metrics = {
                    "loss": float("nan"),
                    "group_loss": float("nan"),
                    "accuracy": float("nan"),
                    "recall_macro": float("nan"),
                    "f1_weighted": float("nan"),
                    "f1_macro": float("nan"),
                }
                val_f1 = train_metrics["f1_macro"]

            current_lr = float(phase2_optimizer.param_groups[0]["lr"])
            if use_group_task:
                logger.info(
                    "Phase2 Epoch %03d | lr=%.6f | train_loss=%.4f (grp=%.4f) train_f1=%.4f | "
                    "val_loss=%.4f (grp=%.4f) val_f1=%.4f",
                    epoch,
                    current_lr,
                    train_metrics["loss"],
                    train_metrics["group_loss"],
                    train_metrics["f1_macro"],
                    val_metrics["loss"],
                    val_metrics["group_loss"],
                    val_metrics["f1_macro"],
                )
            else:
                logger.info(
                    "Phase2 Epoch %03d | lr=%.6f | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
                    epoch,
                    current_lr,
                    train_metrics["loss"],
                    train_metrics["f1_macro"],
                    val_metrics["loss"],
                    val_metrics["f1_macro"],
                )

            phase2_history.append(
                {
                    "epoch": float(epoch),
                    "lr": current_lr,
                    "train_loss": train_metrics["loss"],
                    "train_group_loss": train_metrics["group_loss"],
                    "train_acc": train_metrics["accuracy"],
                    "train_recall_macro": train_metrics["recall_macro"],
                    "train_f1_weighted": train_metrics["f1_weighted"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "val_loss": val_metrics["loss"],
                    "val_group_loss": val_metrics["group_loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
            )

            if val_f1 > phase2_best_val_f1:
                phase2_best_val_f1 = val_f1
                save_checkpoint(
                    path=phase2_best_model_path,
                    model=model,
                    optimizer=phase2_optimizer,
                    scheduler=phase2_scheduler,
                    epoch=epoch,
                    best_metric=phase2_best_val_f1,
                    metadata={
                        "label_names": prepared.label_names,
                        "group_label_names": prepared.group_label_names if use_group_task else [],
                        "feature_names": prepared.feature_names,
                        "pooling": cfg.model.pooling,
                        "reliability_aware": bool(cfg.model.reliability_aware),
                        "loss_type": cfg.train.loss_type,
                        "focal_gamma": cfg.train.focal_gamma,
                        "logit_adjust_tau": cfg.train.logit_adjust_tau,
                        "class_weight_power": cfg.train.class_weight_power,
                        "num_group_classes": prepared.num_group_classes if use_group_task else 0,
                        "use_group_task": bool(use_group_task),
                        "stage": "phase2_rare_finetune",
                        "phase2_learning_rate": cfg.train.phase2_learning_rate,
                        "rare_class_ids": rare_class_ids.tolist(),
                        "hierarchical_constraint": bool(cfg.train.hierarchical_constraint),
                        "hierarchical_constraint_weight": cfg.train.hierarchical_constraint_weight,
                        "hierarchical_constraint_eps": cfg.train.hierarchical_constraint_eps,
                        "class_group_compat": class_group_compat_np.tolist() if class_group_compat_np is not None else None,
                    },
                )
                logger.info("Saved new phase 2 best model (val_f1=%.4f).", phase2_best_val_f1)

            if phase2_early_stopping is not None and phase2_early_stopping.step(val_f1):
                logger.info("Phase 2 early stopping triggered at epoch %d.", epoch)
                break

        pd.DataFrame(phase2_history).to_csv(run_dir / "history_phase2.csv", index=False)

        if phase2_best_model_path.exists() and phase2_best_val_f1 > best_val_f1:
            best_val_f1 = phase2_best_val_f1
            best_model_path.write_bytes(phase2_best_model_path.read_bytes())
            logger.info(
                "Phase 2 improved best val_f1 to %.4f. Promoted %s -> %s",
                best_val_f1,
                phase2_best_model_path.name,
                best_model_path.name,
            )
        else:
            logger.info(
                "Phase 2 did not improve over phase 1 best val_f1=%.4f. Keeping phase 1 checkpoint.",
                best_val_f1,
            )

    save_epoch_metric_curves(history=history, phase2_history=phase2_history, run_dir=run_dir)

    logger.info("Training finished. Loading best checkpoint from %s", best_model_path)
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        train_sampler=None,
        train_augmentation=None,
    )
    split_evals: dict[str, dict] = {}

    test_eval = evaluate_split(
        model=model,
        dataloader=eval_dataloaders["test"],
        device=device,
        label_names=prepared.label_names,
        pooling=cfg.model.pooling,
        return_attention=True,
    )
    save_eval_artifacts("test", test_eval, prepared.label_names, run_dir, prepared.time_grid)
    split_evals["test"] = test_eval
    logger.info(
        "TEST | exactitude=%.4f precision_macro=%.4f rappel_macro=%.4f f1_macro=%.4f f1_pondere=%.4f",
        test_eval["accuracy"],
        test_eval["precision_macro"],
        test_eval["recall_macro"],
        test_eval["f1_macro"],
        test_eval["f1_weighted"],
    )

    save_split_comparison_artifacts(split_evals, run_dir)
    logger.info("Artefacts de test sauvegardes dans %s", run_dir)


if __name__ == "__main__":
    main()
