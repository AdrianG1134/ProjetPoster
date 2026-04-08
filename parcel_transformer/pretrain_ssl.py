from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm import tqdm

from config import ExperimentConfig, get_default_config
from data import build_dataloaders, prepare_dataset, standardize_prepared_features
from model import TemporalTransformerClassifier
from utils import EarlyStopping, create_run_dir, resolve_device, save_checkpoint, save_json, set_seed, setup_logger


class SSLPretrainModel(nn.Module):
    def __init__(self, backbone: TemporalTransformerClassifier) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.feature_proj.out_features
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, backbone.input_dim),
        )

    def forward(
        self,
        features: Tensor,
        day_of_year: Tensor,
        observed_mask: Tensor,
        quality_features: Tensor,
    ) -> Tensor:
        encoded = self.backbone.encode(
            features=features,
            day_of_year=day_of_year,
            observed_mask=observed_mask,
            quality_features=quality_features,
            return_attention=False,
        )
        temporal_tokens = encoded["temporal_tokens"]
        return self.reconstruction_head(temporal_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised masked pretraining for parcel temporal transformer.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to long CSV input.")
    parser.add_argument("--prepared-npz", type=str, default=None, help="Prepared NPZ input path.")
    parser.add_argument("--save-prepared-npz", type=str, default=None, help="Optional output NPZ path.")
    parser.add_argument("--output-dir", type=str, default="outputs_transformer/ssl_pretrain")

    parser.add_argument("--split-method", type=str, choices=["parcel", "tile"], default=None)
    parser.add_argument("--time-grid-frequency", type=str, default=None)
    parser.add_argument("--index-filter", type=str, default=None)
    parser.add_argument("--min-obs", type=int, default=None)
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
        help="Enable reliability-aware encoder during SSL pretraining.",
    )
    parser.add_argument(
        "--no-reliability-aware",
        dest="reliability_aware",
        action="store_false",
        default=None,
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scheduler", type=str, choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=float, default=0.25, help="Fraction of observed timestamps masked.")
    parser.add_argument("--ssl-loss", type=str, choices=["mse", "l1"], default="mse")
    parser.add_argument(
        "--standardize-features",
        dest="standardize_features",
        action="store_true",
        default=True,
        help="Apply train-only feature standardization before SSL training.",
    )
    parser.add_argument(
        "--no-standardize-features",
        dest="standardize_features",
        action="store_false",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
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

    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.num_workers = args.num_workers
    cfg.train.scheduler = args.scheduler
    cfg.train.scheduler_patience = args.scheduler_patience
    cfg.train.scheduler_factor = args.scheduler_factor
    cfg.train.min_learning_rate = args.min_learning_rate
    cfg.train.early_stopping_patience = args.early_stopping_patience
    cfg.train.gradient_clip_norm = args.gradient_clip_norm
    cfg.train.standardize_features = args.standardize_features
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    return cfg


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
):
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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


def sample_ssl_mask(observed_mask: Tensor, mask_ratio: float) -> Tensor:
    if mask_ratio <= 0.0:
        return torch.zeros_like(observed_mask, dtype=torch.bool)
    random_scores = torch.rand(observed_mask.shape, device=observed_mask.device)
    ssl_mask = (random_scores < mask_ratio) & observed_mask
    needs_one = (ssl_mask.sum(dim=1) == 0) & (observed_mask.sum(dim=1) > 0)
    sample_ids = torch.where(needs_one)[0]
    for i in sample_ids.tolist():
        observed_idx = torch.where(observed_mask[i])[0]
        if observed_idx.numel() == 0:
            continue
        selected = observed_idx[torch.randint(observed_idx.numel(), (1,), device=observed_idx.device)]
        ssl_mask[i, selected] = True
    return ssl_mask


def run_ssl_epoch(
    model: SSLPretrainModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_ratio: float,
    ssl_loss: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: Optional[float] = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_values = 0
    total_masked_tokens = 0

    iterator = tqdm(dataloader, leave=False, desc="ssl_train" if is_train else "ssl_eval")
    for batch in iterator:
        features = batch["features"].to(device=device, dtype=torch.float32)
        day_of_year = batch["day_of_year"].to(device=device, dtype=torch.float32)
        observed_mask = batch["observed_mask"].to(device=device, dtype=torch.bool)
        quality_features = batch["quality_features"].to(device=device, dtype=torch.float32)

        ssl_mask = sample_ssl_mask(observed_mask, mask_ratio=mask_ratio)
        masked_features = features.clone()
        masked_observed = observed_mask.clone()
        masked_quality = quality_features.clone()

        masked_features[ssl_mask] = 0.0
        masked_observed[ssl_mask] = False
        masked_quality[..., 0] = torch.where(
            masked_observed, masked_quality[..., 0], torch.zeros_like(masked_quality[..., 0])
        )
        masked_quality[..., 1] = torch.where(
            masked_observed, masked_quality[..., 1], torch.ones_like(masked_quality[..., 1])
        )
        masked_quality[..., 2] = torch.where(
            masked_observed, masked_quality[..., 2], torch.zeros_like(masked_quality[..., 2])
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            reconstructed = model(
                features=masked_features,
                day_of_year=day_of_year,
                observed_mask=masked_observed,
                quality_features=masked_quality,
            )
            target_values = features[ssl_mask]
            pred_values = reconstructed[ssl_mask]
            if target_values.numel() == 0:
                loss = reconstructed.sum() * 0.0
                batch_abs = 0.0
                batch_sq = 0.0
                batch_values = 0
                masked_tokens = 0
            else:
                diff = pred_values - target_values
                if ssl_loss == "mse":
                    loss = diff.pow(2).mean()
                elif ssl_loss == "l1":
                    loss = diff.abs().mean()
                else:
                    raise ValueError(f"Unsupported ssl_loss: {ssl_loss}")
                batch_abs = float(diff.abs().sum().item())
                batch_sq = float(diff.pow(2).sum().item())
                batch_values = int(diff.numel())
                masked_tokens = int(ssl_mask.sum().item())

            if is_train:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

        if batch_values > 0:
            total_loss += float(loss.item()) * batch_values
            total_abs += batch_abs
            total_sq += batch_sq
            total_values += batch_values
            total_masked_tokens += masked_tokens

    if total_values == 0:
        return {
            "loss": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "masked_tokens": 0.0,
        }

    return {
        "loss": total_loss / float(total_values),
        "mae": total_abs / float(total_values),
        "rmse": math.sqrt(total_sq / float(total_values)),
        "masked_tokens": float(total_masked_tokens),
    }


def extract_encoder_state_dict(backbone: TemporalTransformerClassifier) -> dict[str, Tensor]:
    allowed_prefixes = (
        "feature_proj.",
        "doy_encoding.",
        "layers.",
        "cls_token",
        "reliability_proj.",
        "reliability_gate_logit",
    )
    return {
        k: v.detach().cpu()
        for k, v in backbone.state_dict().items()
        if any(k == prefix or k.startswith(prefix) for prefix in allowed_prefixes)
    }


def main() -> None:
    args = parse_args()
    cfg = apply_args_to_config(args, get_default_config())

    if args.csv_path is None and args.prepared_npz is None and cfg.data.prepared_npz_path is None:
        raise ValueError("Provide --csv-path or --prepared-npz.")
    if cfg.train.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if cfg.train.learning_rate <= 0:
        raise ValueError("lr must be > 0.")
    if cfg.train.batch_size <= 0:
        raise ValueError("batch-size must be > 0.")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask-ratio must be in (0, 1).")
    if cfg.train.early_stopping_patience <= 0:
        raise ValueError("early-stopping-patience must be > 0.")

    set_seed(cfg.train.seed)
    run_dir = create_run_dir(cfg.data.output_dir, prefix="ssl_pretrain")
    logger = setup_logger(run_dir / "ssl_pretrain.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration loaded.")
    logger.info("Reliability-aware encoder: %s", "enabled" if cfg.model.reliability_aware else "disabled")
    save_json(
        {
            "config": cfg.to_dict(),
            "ssl": {
                "mask_ratio": args.mask_ratio,
                "loss": args.ssl_loss,
            },
        },
        run_dir / "ssl_config.json",
    )

    prepared = prepare_dataset(cfg.data)
    logger.info(
        "Prepared dataset: N=%d, T=%d, F=%d, classes=%d",
        prepared.features.shape[0],
        prepared.seq_len,
        prepared.num_features,
        prepared.num_classes,
    )
    split_sizes = {k: int(v.shape[0]) for k, v in prepared.splits.items()}
    logger.info("Split sizes: %s", split_sizes)
    if split_sizes.get("train", 0) == 0:
        raise ValueError("Train split is empty.")

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

    device = resolve_device(cfg.train.device)
    logger.info("Device: %s", device)
    dataloaders = build_dataloaders(
        prepared=prepared,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = TemporalTransformerClassifier(
        input_dim=prepared.num_features,
        num_classes=prepared.num_classes,
        cfg=cfg.model,
        num_group_classes=0,
    )
    model = SSLPretrainModel(backbone=backbone).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, cfg=cfg)
    early_stopping = (
        EarlyStopping(patience=cfg.train.early_stopping_patience, mode="min")
        if split_sizes.get("val", 0) > 0
        else None
    )

    best_metric = float("inf")
    best_model_path = run_dir / "best_ssl_model.pt"
    best_encoder_path = run_dir / "best_ssl_encoder.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_ssl_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            mask_ratio=args.mask_ratio,
            ssl_loss=args.ssl_loss,
            optimizer=optimizer,
            gradient_clip_norm=cfg.train.gradient_clip_norm,
        )

        if split_sizes.get("val", 0) > 0:
            val_metrics = run_ssl_epoch(
                model=model,
                dataloader=dataloaders["val"],
                device=device,
                mask_ratio=args.mask_ratio,
                ssl_loss=args.ssl_loss,
                optimizer=None,
                gradient_clip_norm=None,
            )
            monitor_metric = val_metrics["loss"]
        else:
            val_metrics = {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan"), "masked_tokens": 0.0}
            monitor_metric = train_metrics["loss"]

        current_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %03d | lr=%.6f | train_loss=%.6f train_rmse=%.6f | val_loss=%.6f val_rmse=%.6f",
            epoch,
            current_lr,
            train_metrics["loss"],
            train_metrics["rmse"],
            val_metrics["loss"],
            val_metrics["rmse"],
        )
        history.append(
            {
                "epoch": float(epoch),
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_masked_tokens": train_metrics["masked_tokens"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_masked_tokens": val_metrics["masked_tokens"],
            }
        )

        if monitor_metric < best_metric:
            best_metric = monitor_metric
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                metadata={
                    "task": "ssl_masked_reconstruction",
                    "mask_ratio": args.mask_ratio,
                    "ssl_loss": args.ssl_loss,
                    "feature_names": prepared.feature_names,
                    "reliability_aware": bool(cfg.model.reliability_aware),
                    "model_config": {
                        "d_model": cfg.model.d_model,
                        "n_heads": cfg.model.n_heads,
                        "n_layers": cfg.model.n_layers,
                        "dim_feedforward": cfg.model.dim_feedforward,
                        "dropout": cfg.model.dropout,
                        "pooling": cfg.model.pooling,
                        "reliability_aware": cfg.model.reliability_aware,
                    },
                },
            )
            torch.save(
                {
                    "encoder_state_dict": extract_encoder_state_dict(model.backbone),
                    "metadata": {
                        "task": "ssl_masked_reconstruction",
                        "mask_ratio": args.mask_ratio,
                        "feature_names": prepared.feature_names,
                        "reliability_aware": bool(cfg.model.reliability_aware),
                    },
                },
                best_encoder_path,
            )
            logger.info("Saved new best SSL checkpoint (loss=%.6f).", best_metric)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_metric)
            else:
                scheduler.step()

        if early_stopping is not None and early_stopping.step(monitor_metric):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    pd.DataFrame(history).to_csv(run_dir / "ssl_history.csv", index=False)
    logger.info("Artifacts saved in %s", run_dir)
    logger.info("Best SSL model: %s", best_model_path)
    logger.info("Best encoder checkpoint for fine-tuning: %s", best_encoder_path)


if __name__ == "__main__":
    main()
