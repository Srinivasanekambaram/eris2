from __future__ import annotations

import copy
import json
import os
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm



class EarlyStopping:

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss: Optional[float] = None
        self.best_weights: Optional[dict] = None
        self.best_epoch: Optional[int] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model: nn.Module) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def collate_skip_none(samples: List[Optional[dict]]):
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    return torch.utils.data.dataloader.default_collate(valid)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
    grad_clip_norm: Optional[float] = 1.0,
    desc: str = "",
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.train(mode=is_train)
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            if batch is None:
                continue
            batch = _move_batch(batch, device)
            targets = batch["ddg"]
            preds = model(batch)

            loss = criterion(preds, targets)
            if torch.isnan(loss):
                warnings.warn("NaN loss encountered; skipping batch.")
                continue

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), np.array([]), np.array([])

    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    mean_loss = total_loss / n_batches
    mse = float(mean_squared_error(targets_np, preds_np))
    rmse = float(np.sqrt(mse))
    if len(preds_np) > 1 and np.std(preds_np) > 0 and np.std(targets_np) > 0:
        pcc = float(pearsonr(preds_np, targets_np)[0])
    else:
        pcc = float("nan")
    return mean_loss, mse, rmse, pcc, preds_np, targets_np


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_dir: str,
    criterion: Optional[Callable] = None,
    max_epochs: int = 50,
    early_stop_patience: int = 5,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip_norm: Optional[float] = 1.0,
) -> Tuple[nn.Module, Dict[str, list]]:
    os.makedirs(log_dir, exist_ok=True)
    if criterion is None:
        criterion = nn.HuberLoss()

    early_stop = EarlyStopping(patience=early_stop_patience, restore_best_weights=True)
    history: Dict[str, list] = {
        "epochs": [],
        "train_loss": [], "train_mse": [], "train_rmse": [], "train_pcc": [],
        "val_loss": [], "val_mse": [], "val_rmse": [], "val_pcc": [],
        "learning_rate": [], "time_elapsed": [],
    }
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_mse, train_rmse, train_pcc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, grad_clip_norm=grad_clip_norm,
            desc=f"Epoch {epoch:02d}/{max_epochs} train",
        )
        val_loss, val_mse, val_rmse, val_pcc, _, _ = _run_epoch(
            model, val_loader, criterion, None, device,
            is_train=False,
            desc=f"Epoch {epoch:02d}/{max_epochs} val",
        )

        if lr_scheduler is not None:
            try:
                lr_scheduler.step(val_loss)
            except TypeError:
                lr_scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - start

        history["epochs"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_mse"].append(float(train_mse))
        history["train_rmse"].append(float(train_rmse))
        history["train_pcc"].append(float(train_pcc))
        history["val_loss"].append(float(val_loss))
        history["val_mse"].append(float(val_mse))
        history["val_rmse"].append(float(val_rmse))
        history["val_pcc"].append(float(val_pcc))
        history["learning_rate"].append(current_lr)
        history["time_elapsed"].append(float(elapsed))

        print(
            f"[epoch {epoch:02d}/{max_epochs}] "
            f"train: loss={train_loss:.4f} mse={train_mse:.4f} pcc={train_pcc:.4f} | "
            f"val: loss={val_loss:.4f} mse={val_mse:.4f} pcc={val_pcc:.4f} | "
            f"lr={current_lr:.2e} elapsed={elapsed:.0f}s"
        )

        prev_best = early_stop.best_epoch
        early_stop.step(val_loss, model, epoch)

        with open(os.path.join(log_dir, "history.json"), "w") as f:
            json.dump({**history, "epoch_of_best": early_stop.best_epoch,
                       "best_val_loss": early_stop.best_loss}, f, indent=2)

        if early_stop.best_epoch == epoch and early_stop.best_weights is not None:
            ckpt_path = os.path.join(log_dir, "best_model.pt")
            tmp_path = ckpt_path + f".tmp.{os.getpid()}"
            torch.save(early_stop.best_weights, tmp_path)
            os.replace(tmp_path, ckpt_path)

        if early_stop.should_stop:
            print(f"Early stopping at epoch {epoch} (best={early_stop.best_epoch}, "
                  f"val_loss={early_stop.best_loss:.4f}).")
            break

    early_stop.restore(model)
    history["epoch_of_best"] = early_stop.best_epoch
    history["best_val_loss"] = float(early_stop.best_loss) if early_stop.best_loss is not None else None
    return model, history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    log_dir: Optional[str] = None,
    criterion: Optional[Callable] = None,
) -> Dict[str, object]:
    if criterion is None:
        criterion = nn.HuberLoss()

    start = time.time()
    test_loss, _, _, _, preds, targets = _run_epoch(
        model, loader, criterion, None, device, is_train=False, desc="eval",
    )

    if preds.size == 0:
        raise RuntimeError("evaluate_model: no valid samples in loader")

    from eris2.metrics import full_benchmark_report
    stats_bundle = full_benchmark_report(
        preds, targets,
        per_fold_pccs=None,
        bootstrap_iters=2000,
        permutation_iters=2000,
    )
    mse = float(mean_squared_error(targets, preds))
    r2 = float(r2_score(targets, preds))

    elapsed = time.time() - start


    if len(preds) >= 20:
        mean_pred = float(np.mean(preds))
        mean_true = float(np.mean(targets))
        if (abs(mean_pred) > 1e-3 and abs(mean_true) > 1e-3
                and np.sign(mean_pred) != np.sign(mean_true)):
            warnings.warn(
                f"Warning: mean predicted ddG ({mean_pred:.3f}) has "
                f"opposite sign to mean true ΔΔG ({mean_true:.3f}) over {len(preds)} samples. "
                f"Check training-vs-inference sign handling.",
                RuntimeWarning,
            )

    result = {
        "test_loss": float(test_loss),
        "mse": mse,
        "r2": r2,
        "n_samples": int(len(preds)),
        "predictions": preds,
        "targets": targets,
        "time_elapsed": float(elapsed),
    }
    result.update(stats_bundle)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "eval_results.json"), "w") as f:
            json.dump({k: v for k, v in result.items() if not isinstance(v, np.ndarray)}, f, indent=2)
        np.savez(os.path.join(log_dir, "eval_predictions.npz"), preds=preds, targets=targets)

    return result


def make_default_lr_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )
