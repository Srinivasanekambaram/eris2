from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    esm_dim: int = 1280
    use_onehot: bool = False
    d_model: int = 256
    n_heads: int = 8
    n_sequence_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    window_len: int = 31
    mutation_position: int = 15

    ca_matrix_size: int = 15
    atom_matrix_size: int = 100

    rsa_dim: int = 1
    angles_dim: int = 2
    hbond_dim: int = 3
    ss_dim: int = 8
    charge_dim: int = 1
    hydro_dim: int = 1
    atom_type_dim: int = 4

    per_residue_hidden: int = 64
    struct_hidden: int = 128
    fc_hidden: int = 512
    fc_layers: int = 3

    @property
    def per_residue_input_dim(self) -> int:
        return (
            self.rsa_dim + self.angles_dim + self.hbond_dim + self.ss_dim
            + self.charge_dim + self.hydro_dim + self.atom_type_dim
        )


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:

        h = self.ln1(x)
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, q_stream: torch.Tensor, kv_stream: torch.Tensor, kv_padding_mask: torch.Tensor
    ) -> torch.Tensor:

        q = self.ln_q(q_stream)
        kv = self.ln_kv(kv_stream)
        attn_out, _ = self.mha(q, kv, kv, key_padding_mask=kv_padding_mask, need_weights=False)
        x = q_stream + self.drop1(attn_out)
        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class DistanceMatrixEncoder(nn.Module):

    def __init__(self, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.attn_score = nn.Conv2d(out_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        x = dist_matrix.unsqueeze(1)
        h = self.conv(x)
        scores = self.attn_score(h)
        B, C, S1, S2 = h.shape
        weights = F.softmax(scores.view(B, 1, S1 * S2), dim=-1)
        pooled = (h.view(B, C, S1 * S2) * weights).sum(dim=-1)
        return self.dropout(pooled)


class PerResidueEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(
        self, combined: torch.Tensor, window_mask: torch.Tensor, center: int
    ) -> torch.Tensor:
        h = self.proj(combined)
        center_tok = h[:, center, :]
        mask = window_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        masked_mean = (h * mask).sum(dim=1) / denom
        return torch.cat([center_tok, masked_mean], dim=-1)


class DDGPredictor(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_proj = nn.Sequential(
            nn.Linear(config.esm_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )


        self.sequence_encoder = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.d_model, n_heads=config.n_heads,
                d_ff=config.d_ff, dropout=config.dropout,
            )
            for _ in range(config.n_sequence_layers)
        ])

        self.cross_wt_from_mut = CrossAttentionBlock(
            d_model=config.d_model, n_heads=config.n_heads,
            d_ff=config.d_ff, dropout=config.dropout,
        )
        self.cross_mut_from_wt = CrossAttentionBlock(
            d_model=config.d_model, n_heads=config.n_heads,
            d_ff=config.d_ff, dropout=config.dropout,
        )

        self.ca_encoder = DistanceMatrixEncoder(
            out_dim=config.struct_hidden, dropout=config.dropout,
        )
        self.atom_encoder = DistanceMatrixEncoder(
            out_dim=config.struct_hidden, dropout=config.dropout,
        )

        self.per_residue_encoder = PerResidueEncoder(
            in_dim=config.per_residue_input_dim,
            hidden=config.per_residue_hidden,
            dropout=config.dropout,
        )

        fc_input_dim = (
            3 * config.d_model
            + 2 * config.struct_hidden
            + 2 * config.per_residue_hidden
        )
        layers = []
        in_dim = fc_input_dim
        for _ in range(config.fc_layers):
            layers.extend([
                nn.Linear(in_dim, config.fc_hidden),
                nn.LayerNorm(config.fc_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.fc_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def _encode_sequence(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        for block in self.sequence_encoder:
            x = block(x, key_padding_mask=key_padding_mask)
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config
        wt_emb = batch["wt_embedding"]
        mut_emb = batch["mut_embedding"]
        window_mask = batch["window_mask"]
        key_padding = (window_mask == 0).bool()

        wt_h = self.embed_proj(wt_emb)
        mut_h = self.embed_proj(mut_emb)

        wt_h = self._encode_sequence(wt_h, key_padding)
        mut_h = self._encode_sequence(mut_h, key_padding)

        wt_refined = self.cross_wt_from_mut(wt_h, mut_h, kv_padding_mask=key_padding)
        mut_refined = self.cross_mut_from_wt(mut_h, wt_h, kv_padding_mask=key_padding)

        wt_center = wt_refined[:, cfg.mutation_position, :]
        mut_center = mut_refined[:, cfg.mutation_position, :]
        diff_center = mut_center - wt_center

        ca_pool = self.ca_encoder(batch["ca_distance_matrix"])
        atom_pool = self.atom_encoder(batch["atom_distance_matrix"])

        per_res = torch.cat([
            batch["rsa_values"].unsqueeze(-1),
            batch["backbone_angles"],
            batch["hbond_features"],
            batch["ss_features"],
            batch["charge_features"],
            batch["hydrophobicity_features"],
            batch["atom_features"],
        ], dim=-1)
        per_res_readout = self.per_residue_encoder(
            per_res, window_mask=window_mask, center=cfg.mutation_position,
        )

        combined = torch.cat([
            wt_center, mut_center, diff_center,
            ca_pool, atom_pool,
            per_res_readout,
        ], dim=-1)
        out = self.regressor(combined).squeeze(-1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
