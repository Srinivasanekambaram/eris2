from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eris2.data import DatasetConfig, ProteinDataset

PDBS = ROOT / "data" / "example" / "pdbs"


def _dataset(cache: Path, pdb_dir: Path) -> ProteinDataset:
    rows = pd.DataFrame({"uniprot": ["2ZTA"], "chain": ["A"],
                         "mut": ["E20Q"], "ddg": [0.0]})
    return ProteinDataset(DatasetConfig(
        csv_file=rows, pdb_folder=str(pdb_dir), cache_dir=str(cache)))


def test_same_contents_give_the_same_key(tmp_path):
    ds = _dataset(tmp_path, PDBS)
    a = ds._protein_cache_path(str(PDBS / "2ZTA.pdb"), "A")
    b = ds._protein_cache_path(str(PDBS / "2ZTA.pdb"), "A")
    assert a is not None and a == b


def test_same_name_different_contents_give_different_keys(tmp_path):
    impostor_dir = tmp_path / "uploaded"
    impostor_dir.mkdir()
    (impostor_dir / "2ZTA.pdb").write_bytes((PDBS / "3G1G.pdb").read_bytes())

    ds = _dataset(tmp_path / "cache", PDBS)
    real = ds._protein_cache_path(str(PDBS / "2ZTA.pdb"), "A")
    fake = ds._protein_cache_path(str(impostor_dir / "2ZTA.pdb"), "A")
    assert real is not None and fake is not None
    assert real != fake, "same filename, different structure -> must not collide"


def test_different_chains_give_different_keys(tmp_path):
    ds = _dataset(tmp_path, PDBS)
    f = str(PDBS / "2ZTA.pdb")
    assert ds._protein_cache_path(f, "A") != ds._protein_cache_path(f, "B")


def test_key_changes_with_the_hbond_cutoff(tmp_path):
    rows = pd.DataFrame({"uniprot": ["2ZTA"], "chain": ["A"],
                         "mut": ["E20Q"], "ddg": [0.0]})
    f = str(PDBS / "2ZTA.pdb")
    keys = set()
    for cutoff in (3.5, 4.0):
        ds = ProteinDataset(DatasetConfig(
            csv_file=rows, pdb_folder=str(PDBS), cache_dir=str(tmp_path),
            hbond_distance_cutoff_a=cutoff))
        keys.add(ds._protein_cache_path(f, "A"))
    assert len(keys) == 2


def test_no_cache_path_when_disk_cache_is_off(tmp_path):
    rows = pd.DataFrame({"uniprot": ["2ZTA"], "chain": ["A"],
                         "mut": ["E20Q"], "ddg": [0.0]})
    ds = ProteinDataset(DatasetConfig(
        csv_file=rows, pdb_folder=str(PDBS), cache_dir=str(tmp_path),
        use_disk_cache=False))
    assert ds._protein_cache_path(str(PDBS / "2ZTA.pdb"), "A") is None


def test_missing_file_does_not_raise(tmp_path):
    ds = _dataset(tmp_path, PDBS)
    assert ds._protein_cache_path(str(tmp_path / "nope.pdb"), "A") is None
