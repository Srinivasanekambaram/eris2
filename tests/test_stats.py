from __future__ import annotations

import numpy as np


def test_pcc_and_scc_sanity():
    from eris2.metrics import pcc_with_stats, scc_with_stats
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = y + 0.3 * rng.normal(size=500)
    r = pcc_with_stats(p, y)
    s = scc_with_stats(p, y)
    for d in (r, s):
        for k in list(d):
            v = d[k]
            if isinstance(v, float):
                assert np.isfinite(v), f"{k}={v}"
    assert 0.8 < r["pcc"] < 1.0
    assert r["pcc_ci95_lo"] < r["pcc"] < r["pcc_ci95_hi"]
    assert r["pcc_pvalue"] < 1e-10


def test_rmse_bootstrap_ci_brackets_point():
    from eris2.metrics import rmse_with_bootstrap_ci
    rng = np.random.default_rng(1)
    y = rng.normal(size=300); p = y + rng.normal(size=300, scale=0.5)
    r = rmse_with_bootstrap_ci(p, y, n_boot=500)
    assert r["rmse_ci95_lo"] < r["rmse"] < r["rmse_ci95_hi"]


def test_auprc_permutation_p_low_for_perfect_ranker():
    from eris2.metrics import auprc_with_permutation_p
    rng = np.random.default_rng(2)
    y = rng.choice([-1.0, 1.0], size=200)
    preds = y + rng.normal(size=200, scale=0.2)
    r = auprc_with_permutation_p(preds, y, n_perm=500)
    assert r["auprc"] > 0.9
    assert r["auprc_pvalue"] < 0.02


def test_steigers_z_correct_direction():
    from eris2.metrics import steigers_z_dependent_pcc
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    a = y + 0.5 * rng.normal(size=300)
    b = rng.normal(size=300)
    r = steigers_z_dependent_pcc(a, b, y)
    assert r["delta_pcc"] > 0.5
    assert r["steiger_pvalue"] < 1e-10


def test_paired_bootstrap_delta_sign():
    from eris2.metrics import paired_bootstrap_delta
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    a = y + 0.3 * rng.normal(size=200)
    b = y + 0.9 * rng.normal(size=200)
    r = paired_bootstrap_delta(a, b, y, metric="pcc", n_boot=500)
    assert r["delta_obs"] > 0
    assert r["delta_ci95_lo"] > 0
    assert r["bootstrap_pvalue"] < 0.05


def test_full_benchmark_report_has_all_keys():
    from eris2.metrics import full_benchmark_report
    rng = np.random.default_rng(5)
    y = rng.normal(size=200); p = y + rng.normal(size=200, scale=0.5)
    r = full_benchmark_report(p, y, per_fold_pccs=[0.6, 0.5, 0.55],
                              bootstrap_iters=200, permutation_iters=200)
    for k in ("pcc", "pcc_pvalue", "pcc_ci95_lo", "pcc_ci95_hi",
              "scc", "scc_pvalue",
              "rmse", "rmse_ci95_lo", "rmse_ci95_hi",
              "auprc", "auprc_pvalue", "prevalence", "n_positive",
              "per_fold_pcc_mean", "per_fold_pcc_sd", "per_fold_n",
              "n"):
        assert k in r, f"missing key {k}"
