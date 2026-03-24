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

        self.pooling = cfg.pooling
        self.feature_proj = nn.Linear(input_dim, cfg.d_model)
        self.doy_encoding = DayOfYearEncoding(cfg.d_model, cfg.dropout)
        self.input_dropout = nn.Dropout(cfg.dropout)
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

    def forward(
        self,
        features: Tensor,
        day_of_year: Tensor,
        observed_mask: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        """
        Args:
            features: [B, T, F]
            day_of_year: [B, T]
            observed_mask: [B, T] with True where date exists.
        """
        key_padding_mask = ~observed_mask.bool()  # True means "ignore token"

        x = self.feature_proj(features) + self.doy_encoding(day_of_year)
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

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            valid_tokens = (~key_padding_mask).unsqueeze(-1).float()
            denominator = valid_tokens.sum(dim=1).clamp(min=1.0)
            pooled = (x * valid_tokens).sum(dim=1) / denominator

        logits = self.classifier(pooled)
        outputs: dict[str, Tensor | list[Tensor]] = {"logits": logits}
        if self.group_classifier is not None:
            outputs["group_logits"] = self.group_classifier(pooled)
        if return_attention:
            outputs["attention_maps"] = attentions
        return outputs
