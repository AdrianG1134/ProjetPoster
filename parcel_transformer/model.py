from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from config import ModelConfig


class DayOfYearEncoding(nn.Module):
    """Fourier day-of-year encoding projected into model space."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, day_of_year: Tensor) -> Tensor:
        # day_of_year: [B, T] in [1, 366]
        doy = day_of_year.clamp(min=1.0, max=366.0)
        angle = 2.0 * math.pi * (doy / 366.0)
        sinusoidal = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(sinusoidal)


class TemporalEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        norm_first: bool,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.norm_first = norm_first

    def _self_attn_block(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor],
        return_attention: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        return self.dropout1(attn_output), attn_weights if return_attention else None

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.norm_first:
            attn_out, attn_weights = self._self_attn_block(
                self.norm1(x), key_padding_mask=key_padding_mask, return_attention=return_attention
            )
            x = x + attn_out
            x = x + self._ff_block(self.norm2(x))
            return x, attn_weights

        attn_out, attn_weights = self._self_attn_block(
            x, key_padding_mask=key_padding_mask, return_attention=return_attention
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self._ff_block(x))
        return x, attn_weights


class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cfg: ModelConfig,
        num_group_classes: int = 0,
    ) -> None:
        super().__init__()
        if cfg.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {cfg.pooling}")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_group_classes = num_group_classes
        self.pooling = cfg.pooling
        self.reliability_aware = bool(getattr(cfg, "reliability_aware", False))
        self.feature_proj = nn.Linear(input_dim, cfg.d_model)
        self.doy_encoding = DayOfYearEncoding(cfg.d_model, cfg.dropout)
        self.input_dropout = nn.Dropout(cfg.dropout)
        if self.reliability_aware:
            self.reliability_proj = nn.Sequential(
                nn.Linear(3, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.reliability_dropout = nn.Dropout(cfg.dropout)
            self.reliability_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.reliability_proj = None
            self.reliability_dropout = None
            self.register_parameter("reliability_gate_logit", None)
        self.layers = nn.ModuleList(
            [
                TemporalEncoderLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    norm_first=cfg.use_layer_norm_first,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, num_classes),
        )
        self.group_classifier = (
            nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, num_group_classes),
            )
            if num_group_classes > 0
            else None
        )
        if num_group_classes > 0:
            compat_default = torch.ones((num_group_classes, num_classes), dtype=torch.float32)
        else:
            compat_default = torch.empty((0, 0), dtype=torch.float32)

        # Non-persistent buffers: moved with .to(device), but not saved in checkpoints.
        # This keeps backward compatibility with older checkpoints.
        self.register_buffer("class_group_compat", compat_default, persistent=False)
        self.register_buffer("hierarchical_constraint_enabled", torch.tensor(False), persistent=False)
        self.register_buffer("hierarchical_constraint_weight", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("hierarchical_constraint_eps", torch.tensor(1.0e-6, dtype=torch.float32), persistent=False)

    def _prepare_quality_inputs(
        self,
        observed_mask: Tensor,
        quality_features: Optional[Tensor],
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        observed = observed_mask.float()
        if quality_features is None:
            cloud = 1.0 - observed
            px = observed
            quality = torch.stack([observed, cloud, px], dim=-1)
        else:
            quality = quality_features.to(device=observed_mask.device, dtype=dtype)
            if quality.ndim != 3:
                raise ValueError("quality_features must have shape [B, T, Q].")
            if quality.shape[0] != observed_mask.shape[0] or quality.shape[1] != observed_mask.shape[1]:
                raise ValueError(
                    "quality_features must match features/observed_mask shape on the first two dimensions."
                )
            if quality.shape[-1] < 3:
                pad = torch.zeros(
                    quality.shape[0],
                    quality.shape[1],
                    3 - quality.shape[-1],
                    device=quality.device,
                    dtype=quality.dtype,
                )
                quality = torch.cat([quality, pad], dim=-1)
            elif quality.shape[-1] > 3:
                quality = quality[..., :3]

        quality = quality.clone()
        quality[..., 1] = torch.where(observed_mask, quality[..., 1].clamp(0.0, 1.0), torch.ones_like(quality[..., 1]))
        quality[..., 2] = torch.where(observed_mask, quality[..., 2].clamp(0.0, 1.0), torch.zeros_like(quality[..., 2]))
        reliability = torch.where(observed_mask, quality[..., 0].clamp(0.0, 1.0), torch.zeros_like(quality[..., 0]))
        quality[..., 0] = reliability
        return quality, reliability

    def encode(
        self,
        features: Tensor,
        day_of_year: Tensor,
        observed_mask: Tensor,
        quality_features: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        key_padding_mask = ~observed_mask.bool()  # True means "ignore token"

        x = self.feature_proj(features) + self.doy_encoding(day_of_year)
        temporal_reliability = observed_mask.float()
        if self.reliability_aware:
            quality_inputs, temporal_reliability = self._prepare_quality_inputs(
                observed_mask=observed_mask.bool(),
                quality_features=quality_features,
                dtype=features.dtype,
            )
            reliability_embed = self.reliability_proj(quality_inputs)
            if self.reliability_dropout is not None:
                reliability_embed = self.reliability_dropout(reliability_embed)
            gate_strength = torch.sigmoid(self.reliability_gate_logit)
            x = x + reliability_embed
            token_gate = (1.0 - gate_strength) + (gate_strength * temporal_reliability.unsqueeze(-1))
            x = x * token_gate
        x = self.input_dropout(x)

        if self.pooling == "cls":
            cls = self.cls_token.expand(features.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_pad = torch.zeros((features.shape[0], 1), dtype=torch.bool, device=features.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)

        attentions: list[Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(
                x,
                key_padding_mask=key_padding_mask,
                return_attention=return_attention,
            )
            if return_attention and attn_weights is not None:
                attentions.append(attn_weights)

        temporal_tokens = x[:, 1:] if self.pooling == "cls" else x
        return {
            "encoded_tokens": x,
            "temporal_tokens": temporal_tokens,
            "key_padding_mask": key_padding_mask,
            "temporal_reliability": temporal_reliability,
            "attention_maps": attentions,
        }

    def configure_hierarchical_constraint(
        self,
        class_group_compat: Tensor,
        weight: float,
        eps: float = 1.0e-6,
        enabled: bool = True,
    ) -> None:
        if self.num_group_classes <= 0:
            raise ValueError("Cannot enable hierarchical constraint without group head.")
        if class_group_compat.ndim != 2:
            raise ValueError("class_group_compat must be a 2D tensor [G, C].")
        if class_group_compat.shape[0] != self.num_group_classes:
            raise ValueError(
                f"class_group_compat has invalid group dimension: {class_group_compat.shape[0]} "
                f"(expected {self.num_group_classes})."
            )
        if class_group_compat.shape[1] != self.num_classes:
            raise ValueError(
                f"class_group_compat has invalid class dimension: {class_group_compat.shape[1]} "
                f"(expected {self.num_classes})."
            )
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        compat = class_group_compat.to(device=self.class_group_compat.device, dtype=torch.float32)
        compat = compat.clamp(min=0.0)
        col_sum = compat.sum(dim=0, keepdim=True)
        zero_cols = col_sum <= 0.0
        if torch.any(zero_cols):
            compat[:, zero_cols.squeeze(0)] = 1.0 / float(self.num_group_classes)
            col_sum = compat.sum(dim=0, keepdim=True)
        compat = compat / col_sum.clamp(min=1.0e-12)

        self.class_group_compat.copy_(compat)
        self.hierarchical_constraint_weight.fill_(float(weight))
        self.hierarchical_constraint_eps.fill_(float(eps))
        self.hierarchical_constraint_enabled.fill_(bool(enabled))

    def forward(
        self,
        features: Tensor,
        day_of_year: Tensor,
        observed_mask: Tensor,
        quality_features: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        """
        Args:
            features: [B, T, F]
            day_of_year: [B, T]
            observed_mask: [B, T] with True where date exists.
        """
        encoded = self.encode(
            features=features,
            day_of_year=day_of_year,
            observed_mask=observed_mask,
            quality_features=quality_features,
            return_attention=return_attention,
        )
        x = encoded["encoded_tokens"]
        key_padding_mask = encoded["key_padding_mask"]
        temporal_reliability = encoded["temporal_reliability"]
        attentions = encoded["attention_maps"]

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            valid_tokens = (~key_padding_mask).unsqueeze(-1).float()
            if self.reliability_aware:
                rel_weights = temporal_reliability.unsqueeze(-1).clamp(min=0.05)
                token_weights = valid_tokens * rel_weights
            else:
                token_weights = valid_tokens
            denominator = token_weights.sum(dim=1).clamp(min=1.0)
            pooled = (x * token_weights).sum(dim=1) / denominator

        logits = self.classifier(pooled)
        outputs: dict[str, Tensor | list[Tensor]] = {"logits": logits}
        if self.group_classifier is not None:
            group_logits = self.group_classifier(pooled)
            outputs["group_logits"] = group_logits

            if bool(self.hierarchical_constraint_enabled.item()):
                group_probs = torch.softmax(group_logits, dim=1)
                class_compat = torch.matmul(group_probs, self.class_group_compat)
                constrained_logits = logits + float(self.hierarchical_constraint_weight.item()) * torch.log(
                    class_compat.clamp(min=float(self.hierarchical_constraint_eps.item()))
                )
                outputs["logits_raw"] = logits
                outputs["logits"] = constrained_logits
                outputs["hierarchical_class_compat"] = class_compat
        if self.reliability_aware:
            outputs["temporal_reliability"] = temporal_reliability
        if return_attention:
            outputs["attention_maps"] = attentions
        return outputs
