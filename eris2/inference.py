from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from eris2.model import DDGPredictor, ModelConfig


def collate(samples):
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    return torch.utils.data.dataloader.default_collate(valid)


def load_model_config(path: str | None) -> ModelConfig:
    if path is None:
        return ModelConfig()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ModelConfig(**cfg.get("model", {}))


BUNDLE_FORMAT = "eris2-ensemble-v1"


def _is_bundle(obj) -> bool:
    return isinstance(obj, dict) and obj.get("format") == BUNDLE_FORMAT


def load_models(checkpoints, config: str | None, device) -> list:
    if isinstance(checkpoints, (str, Path)):
        checkpoints = [checkpoints]
    mconfig = load_model_config(config)
    models = []
    for ckpt in checkpoints:
        if not Path(ckpt).exists():
            raise SystemExit(
                f"Checkpoint not found: {ckpt}\n\n"
                f"The trained weights are distributed separately from the code.\n"
                f"Download eris2_ensemble.pt into model/, or pass --checkpoint\n"
                f"with its location. In Docker, mount it:\n"
                f"    -v /path/to/eris2_ensemble.pt:/opt/eris2/model/eris2_ensemble.pt")
        try:
            blob = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
        state_dicts = blob["state_dicts"] if _is_bundle(blob) else [blob]
        for sd in state_dicts:
            m = DDGPredictor(mconfig).to(device)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
    return models


def predict_batch(models: Sequence, loader: DataLoader, device):
    per_model, labels, rows = [], None, None
    for i, m in enumerate(models):
        preds, lab, idx = [], [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                preds.extend(m(batch).cpu().numpy().tolist())
                if i == 0:
                    lab.extend(batch["ddg"].cpu().numpy().tolist())
                    if "row_index" in batch:
                        idx.extend(batch["row_index"].cpu().numpy().tolist())
        per_model.append(np.asarray(preds, dtype=float))
        if i == 0:
            labels = np.asarray(lab, dtype=float)
            rows = np.asarray(idx, dtype=int) if idx else None

    lengths = {len(p) for p in per_model}
    if len(lengths) != 1:
        raise RuntimeError(
            f"checkpoints returned different numbers of predictions ({sorted(lengths)}). "
            f"This means the dataset dropped different rows on different passes; "
            f"predictions cannot be aligned.")

    stacked = np.stack(per_model, axis=0)
    if rows is None:
        rows = np.arange(stacked.shape[1], dtype=int)
    elif len(rows) != stacked.shape[1]:
        raise RuntimeError(
            f"got {len(rows)} row indices for {stacked.shape[1]} predictions; "
            f"the dataset is not reporting row_index consistently.")
    return stacked.mean(axis=0), stacked, labels, rows
