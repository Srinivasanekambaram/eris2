from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2.data import MUTATION_RE, _dssp_input
from eris2.inference import collate, predict_batch


class _StubModel:

    def __call__(self, batch):
        return batch["row_index"].float() * 10.0

    def eval(self):
        return self


def _batches(rows, batch_size=2):
    samples = [{"ddg": torch.tensor(float(i)),
                "row_index": torch.tensor(int(i))} for i in rows]
    out = []
    for i in range(0, len(samples), batch_size):
        out.append(collate(samples[i:i + batch_size]))
    return out


def test_predict_batch_reports_surviving_row_indices():
    survivors = [0, 1, 3, 4, 7]
    ddg, per_model, labels, rows = predict_batch(
        [_StubModel()], _batches(survivors), device=torch.device("cpu"))

    assert list(rows) == survivors
    assert len(ddg) == len(survivors)
    np.testing.assert_allclose(ddg, np.array(survivors) * 10.0)


def test_labels_join_on_row_index_not_position():
    df = pd.DataFrame({
        "uniprot": [f"P{i}" for i in range(8)],
        "chain": ["A"] * 8,
        "mut": [f"A{i + 1}G" for i in range(8)],
    })
    survivors = [0, 1, 3, 4, 7]
    _, _, _, rows = predict_batch([_StubModel()], _batches(survivors),
                                  device=torch.device("cpu"))

    joined = df.iloc[rows]
    assert list(joined["uniprot"]) == ["P0", "P1", "P3", "P4", "P7"]
    assert list(joined["uniprot"]) != list(df.iloc[:len(rows)]["uniprot"])


def test_collate_drops_none_and_keeps_the_rest():
    samples = [{"ddg": torch.tensor(1.0), "row_index": torch.tensor(0)},
               None,
               {"ddg": torch.tensor(3.0), "row_index": torch.tensor(2)}]
    batch = collate(samples)
    assert list(batch["row_index"].numpy()) == [0, 2]
    assert collate([None, None]) is None


@pytest.mark.parametrize("mut", ["C191F", "A1G", "W12345Y"])
def test_mutation_regex_accepts_well_formed(mut):
    assert MUTATION_RE.match(mut)


@pytest.mark.parametrize("mut", ["XYZ", "", "A1", "1AG", "AXG", "B12D", "A12Z"])
def test_mutation_regex_rejects_malformed(mut):
    assert not MUTATION_RE.match(mut)


def test_dssp_input_prepends_header_when_missing(tmp_path):
    p = tmp_path / "x.pdb"
    p.write_text("ATOM      1  N   MET A   1      10.0  21.0   9.0  1.00 19.0\n")

    fixed = Path(_dssp_input(str(p)))
    assert fixed != p
    lines = fixed.read_text().splitlines()
    assert lines[0].startswith("HEADER")
    assert lines[1].startswith("ATOM")


def test_dssp_input_leaves_headed_files_alone(tmp_path):
    p = tmp_path / "y.pdb"
    p.write_text("HEADER    PROTEIN\nATOM      1  N   MET A   1     "
                 " 10.0  21.0   9.0  1.00 19.0\n")
    assert _dssp_input(str(p)) == str(p)
