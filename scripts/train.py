from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dataset_checking import DatasetConfig, ProteinDataset
from model import DDGPredictor, ModelConfig, count_parameters
from mut_seq import (
    collate_skip_none, evaluate_model, make_default_lr_scheduler, train_model,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_or_load_folds(
    n_samples: int, n_folds: int, val_fraction: float, split_seed: int,
    folds_path: Path, freeze: bool,
    cluster_labels: "np.ndarray | None" = None,
) -> List[Dict[str, list]]:
    if freeze and folds_path.exists():
        with open(folds_path) as f:
            folds = json.load(f)
        if len(folds) != n_folds:
            raise RuntimeError(f"{folds_path} has {len(folds)} folds, expected {n_folds}")
        for k, f in enumerate(folds):
            union = set(f["train"]) | set(f["val"]) | set(f["test"])
            if len(union) != n_samples:
                raise RuntimeError(
                    f"fold {k+1}: train+val+test covers {len(union)} indices "
                    f"but dataset has {n_samples}"
                )
        if cluster_labels is not None:
            _validate_cluster_disjoint(folds, cluster_labels)
        print(f"[folds] loaded frozen split from {folds_path}")
        return folds

    if cluster_labels is not None:
        assert len(cluster_labels) == n_samples, \
            f"cluster_labels len ({len(cluster_labels)}) != n_samples ({n_samples})"
        print(f"[folds] generating STRICT cluster-aware {n_folds}-fold split "
              f"({len(set(cluster_labels))} clusters, seed={split_seed})")
        folds = _make_cluster_aware_folds(
            cluster_labels=cluster_labels, n_folds=n_folds,
            val_fraction=val_fraction, split_seed=split_seed,
        )
    else:
        print(f"[folds] generating {n_folds}-fold RANDOM split (seed={split_seed})")
        folds = _make_random_folds(n_samples, n_folds, val_fraction, split_seed)

    if freeze:
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = folds_path.with_suffix(folds_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(folds, f, indent=2)
        os.replace(tmp, folds_path)
        print(f"[folds] wrote {folds_path}")
    return folds


def _make_random_folds(n_samples, n_folds, val_fraction, split_seed):
    rng = np.random.default_rng(split_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_size = n_samples // n_folds
    fold_slices = [
        indices[k * fold_size : (k + 1) * fold_size].tolist()
        for k in range(n_folds - 1)
    ]
    fold_slices.append(indices[(n_folds - 1) * fold_size :].tolist())

    folds = []
    for k in range(n_folds):
        test = fold_slices[k]
        remaining = np.array([i for j, sl in enumerate(fold_slices) if j != k for i in sl])
        rng2 = np.random.default_rng(split_seed + k + 1)
        rng2.shuffle(remaining)
        n_val = max(1, int(round(len(remaining) * val_fraction)))
        val = remaining[:n_val].tolist()
        train = remaining[n_val:].tolist()
        folds.append({"train": sorted(train), "val": sorted(val), "test": sorted(test)})
    return folds


def _make_cluster_aware_folds(cluster_labels, n_folds, val_fraction, split_seed):
    n_samples = len(cluster_labels)
    labels_arr = np.asarray(cluster_labels)
    unique_clusters = np.array(sorted(set(labels_arr)))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(unique_clusters)

    cluster_to_test_fold = {c: i % n_folds for i, c in enumerate(unique_clusters)}
    fold_rows = {k: [] for k in range(n_folds)}
    for row_idx, c in enumerate(labels_arr):
        fold_rows[cluster_to_test_fold[c]].append(row_idx)

    folds = []
    for k in range(n_folds):
        test_indices = fold_rows[k]
        remaining_clusters = [c for c in unique_clusters if cluster_to_test_fold[c] != k]
        rng2 = np.random.default_rng(split_seed + k + 1)
        rng2.shuffle(remaining_clusters)

        non_test_row_count = n_samples - len(test_indices)
        val_target = max(1, int(round(non_test_row_count * val_fraction)))
        val_clusters = set()
        val_indices = []
        for c in remaining_clusters:
            if len(val_indices) >= val_target:
                break
            member_rows = [row_idx for row_idx, cl in enumerate(labels_arr) if cl == c]
            val_clusters.add(c)
            val_indices.extend(member_rows)

        train_indices = [
            row_idx for row_idx, cl in enumerate(labels_arr)
            if cluster_to_test_fold[cl] != k and cl not in val_clusters
        ]
        folds.append({
            "train": sorted(train_indices),
            "val":   sorted(val_indices),
            "test":  sorted(test_indices),
        })
    return folds


def _validate_cluster_disjoint(folds, cluster_labels):
    labels_arr = np.asarray(cluster_labels)
    for k, f in enumerate(folds):
        c_train = set(labels_arr[np.array(f["train"], dtype=np.int64)]) if f["train"] else set()
        c_val   = set(labels_arr[np.array(f["val"],   dtype=np.int64)]) if f["val"]   else set()
        c_test  = set(labels_arr[np.array(f["test"],  dtype=np.int64)]) if f["test"]  else set()
        assert c_train.isdisjoint(c_val), f"fold {k+1}: train and val share clusters"
        assert c_train.isdisjoint(c_test), f"fold {k+1}: train and test share clusters"
        assert c_val.isdisjoint(c_test), f"fold {k+1}: val and test share clusters"


def _build_loaders(
    dataset: ProteinDataset, fold: Dict[str, list], batch_size: int, num_workers: int,
) -> Dict[str, DataLoader]:
    common = dict(
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_skip_none, pin_memory=torch.cuda.is_available(),
    )
    return {
        "train": DataLoader(Subset(dataset, fold["train"]), shuffle=True, **common),
        "val":   DataLoader(Subset(dataset, fold["val"]),   shuffle=False, **common),
        "test":  DataLoader(Subset(dataset, fold["test"]),  shuffle=False, **common),
    }


def _run_one_fold(
    fold_idx: int, fold: Dict[str, list], dataset: ProteinDataset, config: dict,
    experiment_dir: Path, device: torch.device,
) -> Dict[str, float]:
    fold_dir = experiment_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    loaders = _build_loaders(
        dataset, fold, batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )
    print(
        f"\n===== FOLD {fold_idx} =====\n"
        f"  train: {len(loaders['train'].dataset)} samples\n"
        f"  val:   {len(loaders['val'].dataset)} samples\n"
        f"  test:  {len(loaders['test'].dataset)} samples"
    )

    model = DDGPredictor(ModelConfig(**config["model"])).to(device)
    print(f"  model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
    )
    scheduler = make_default_lr_scheduler(optimizer)

    model, history = train_model(
        model, loaders["train"], loaders["val"],
        optimizer=optimizer, device=device, log_dir=str(fold_dir),
        max_epochs=train_cfg["max_epochs"],
        early_stop_patience=train_cfg["early_stop_patience"],
        lr_scheduler=scheduler,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
    )
    torch.save(model.state_dict(), fold_dir / "best_model.pt")

    eval_out = evaluate_model(model, loaders["test"], device=device, log_dir=str(fold_dir))
    print(
        f"[fold {fold_idx}] test: mse={eval_out['mse']:.4f} rmse={eval_out['rmse']:.4f} "
        f"pcc={eval_out['pcc']:.4f} scc={eval_out['scc']:.4f} r2={eval_out['r2']:.4f} "
        f"(n={eval_out['n_samples']})"
    )
    return {
        "fold": fold_idx,
        "test_mse": eval_out["mse"], "test_rmse": eval_out["rmse"],
        "test_pcc": eval_out["pcc"], "test_scc": eval_out["scc"], "test_r2": eval_out["r2"],
        "test_n": eval_out["n_samples"],
        "best_epoch": history.get("epoch_of_best"),
        "best_val_loss": history.get("best_val_loss"),
    }


def _write_summary(per_fold: List[Dict[str, float]], out_path: Path) -> None:
    def _agg(key: str) -> Dict[str, float]:
        vals = [f[key] for f in per_fold if f[key] is not None]
        return {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    summary = {
        "per_fold": per_fold,
        "aggregate": {
            "test_mse":  _agg("test_mse"),
            "test_rmse": _agg("test_rmse"),
            "test_pcc":  _agg("test_pcc"),
            "test_scc":  _agg("test_scc"),
            "test_r2":   _agg("test_r2"),
        },
        "n_folds": len(per_fold),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] wrote {out_path}")
    a = summary["aggregate"]
    print(
        f"[summary] test_mse = {a['test_mse']['mean']:.4f} ± {a['test_mse']['std']:.4f}  "
        f"(N={len(per_fold)} folds)"
    )
    print(
        f"[summary] test_pcc = {a['test_pcc']['mean']:.4f} ± {a['test_pcc']['std']:.4f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--fold", type=int, default=None,
                    help="if set, only train this fold (1-indexed). Useful for parallel SLURM.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    experiment_dir = Path(exp["output_dir"]) / exp["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, experiment_dir / "config.yaml")
    print(f"[main] experiment dir: {experiment_dir}")

    _set_all_seeds(exp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device: {device}")

    ds_cfg = DatasetConfig(**config["dataset"])
    dataset = ProteinDataset(ds_cfg)
    print(f"[main] dataset: {len(dataset)} rows loaded from {ds_cfg.csv_file}")

    split = config["splitting"]
    cluster_col = split.get("cluster_column")
    allow_random = bool(split.get("allow_random_split", False))

    if cluster_col:
        if cluster_col not in dataset.data.columns:
            raise ValueError(
                f"splitting.cluster_column='{cluster_col}' not found in CSV columns "
                f"{list(dataset.data.columns)}. Cluster-aware splits are required by "
                f"default. Add the column (recommended) or set "
                f"`splitting.allow_random_split: true` and omit `cluster_column` for "
                f"a random-split run."
            )
        cluster_labels = dataset.data[cluster_col].to_numpy()
        n_clusters = len(set(cluster_labels))
        print(f"[main] using STRICT cluster-aware split — {n_clusters} clusters from column '{cluster_col}'")
    else:
        if not allow_random:
            raise ValueError(
                "splitting.cluster_column is not set. Cluster-aware splits are "
                "required by default to avoid homology leakage. Either add a "
                "cluster label column to your CSV and set `splitting.cluster_column`, "
                "or set `splitting.allow_random_split: true` to acknowledge the "
                "leakage risk and use a random split."
            )
        cluster_labels = None
        print("[main] WARNING: cluster_column not set and allow_random_split=true — "
              "using RANDOM split. Homology leakage is possible.")

    folds = build_or_load_folds(
        n_samples=len(dataset), n_folds=split["n_folds"],
        val_fraction=split["val_fraction"], split_seed=split["split_seed"],
        folds_path=experiment_dir / "folds.json", freeze=split.get("freeze_folds", True),
        cluster_labels=cluster_labels,
    )

    if args.fold is not None:
        if not 1 <= args.fold <= len(folds):
            print(f"--fold {args.fold} out of range [1, {len(folds)}]")
            return 2
        fold_indices = [args.fold]
    else:
        fold_indices = list(range(1, len(folds) + 1))

    per_fold_results: List[Dict[str, float]] = []
    start = time.time()
    for k in fold_indices:
        result = _run_one_fold(
            fold_idx=k, fold=folds[k - 1], dataset=dataset, config=config,
            experiment_dir=experiment_dir, device=device,
        )
        per_fold_results.append(result)
        _write_summary(per_fold_results, experiment_dir / "summary.json")

    total = time.time() - start
    print(f"\n[main] all done in {total / 60:.1f} min")
    
    if len(per_fold_results) > 0 and args.fold is None:
        best_fold_result = max(per_fold_results, key=lambda x: x.get("test_pcc", -float('inf')))
        best_fold_idx = best_fold_result["fold"]
        best_pcc = best_fold_result["test_pcc"]
        
        print(f"\n[main] best fold: {best_fold_idx} with test PCC = {best_pcc:.4f}")

        best_model_src = experiment_dir / f"fold_{best_fold_idx}" / "best_model.pt"
        best_model_dst = experiment_dir / "best_model_overall.pth"
        if best_model_src.exists():
            shutil.copy2(best_model_src, best_model_dst)
            print(f"[main] extracted best model to: {best_model_dst}")
        else:
            print(f"[main] WARNING: could not find {best_model_src} to extract")
            
    return 0


if __name__ == "__main__":
    sys.exit(main())
