from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("eris2_train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
sys.modules["eris2_train"] = train
_spec.loader.exec_module(train)


def test_more_folds_than_clusters_is_rejected_clearly(tmp_path):
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    with pytest.raises(SystemExit) as e:
        train.build_or_load_folds(
            n_samples=len(labels), n_folds=5, val_fraction=0.1, split_seed=1,
            folds_path=tmp_path / "folds.json", freeze=True,
            cluster_labels=labels)
    msg = str(e.value)
    assert "4 sequence cluster" in msg
    assert "5-fold" in msg
    assert "n_folds" in msg


def test_exactly_as_many_clusters_as_folds_is_allowed(tmp_path):
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    folds = train.build_or_load_folds(
        n_samples=len(labels), n_folds=4, val_fraction=0.1, split_seed=1,
        folds_path=tmp_path / "folds.json", freeze=True, cluster_labels=labels)
    assert len(folds) == 4
    assert all(len(f["test"]) > 0 for f in folds)


def test_frozen_split_rejects_a_changed_seed(tmp_path):
    labels = np.array([i // 2 for i in range(20)])
    p = tmp_path / "folds.json"
    kw = dict(n_samples=len(labels), n_folds=5, val_fraction=0.1,
              folds_path=p, freeze=True, cluster_labels=labels)

    train.build_or_load_folds(split_seed=1, **kw)
    assert p.exists() and p.with_name("folds.meta.json").exists()

    with pytest.raises(SystemExit) as e:
        train.build_or_load_folds(split_seed=999, **kw)
    assert "split_seed" in str(e.value)
    assert "delete" in str(e.value)


def test_frozen_split_reloads_when_parameters_match(tmp_path):
    labels = np.array([i // 2 for i in range(20)])
    p = tmp_path / "folds.json"
    kw = dict(n_samples=len(labels), n_folds=5, val_fraction=0.1, split_seed=7,
              folds_path=p, freeze=True, cluster_labels=labels)

    first = train.build_or_load_folds(**kw)
    second = train.build_or_load_folds(**kw)
    assert first == second


def test_legacy_folds_without_sidecar_still_load(tmp_path, capsys):
    labels = np.array([i // 2 for i in range(20)])
    p = tmp_path / "folds.json"
    kw = dict(n_samples=len(labels), n_folds=5, val_fraction=0.1, split_seed=3,
              folds_path=p, freeze=True, cluster_labels=labels)

    folds = train.build_or_load_folds(**kw)
    p.with_name("folds.meta.json").unlink()

    reloaded = train.build_or_load_folds(**kw)
    assert reloaded == folds
    assert "cannot be checked" in capsys.readouterr().out
