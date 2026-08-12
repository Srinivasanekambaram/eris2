from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, mean_squared_error



def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return (float(lo), float(hi))


def pcc_with_stats(preds: np.ndarray, targets: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    preds = np.asarray(preds); targets = np.asarray(targets)
    n = int(preds.size)
    if n < 2 or np.std(preds) == 0 or np.std(targets) == 0:
        return dict(pcc=float("nan"), pcc_pvalue=float("nan"),
                    pcc_ci95_lo=float("nan"), pcc_ci95_hi=float("nan"), n=n)
    r, p = stats.pearsonr(preds, targets)
    lo, hi = _fisher_z_ci(r, n, alpha)
    return dict(pcc=float(r), pcc_pvalue=float(p),
                pcc_ci95_lo=lo, pcc_ci95_hi=hi, n=n)


def scc_with_stats(preds: np.ndarray, targets: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    preds = np.asarray(preds); targets = np.asarray(targets)
    n = int(preds.size)
    if n < 2 or np.std(preds) == 0 or np.std(targets) == 0:
        return dict(scc=float("nan"), scc_pvalue=float("nan"),
                    scc_ci95_lo=float("nan"), scc_ci95_hi=float("nan"))
    rho, p = stats.spearmanr(preds, targets)
    lo, hi = _fisher_z_ci(rho, n, alpha)
    return dict(scc=float(rho), scc_pvalue=float(p),
                scc_ci95_lo=lo, scc_ci95_hi=hi)


def rmse_with_bootstrap_ci(
    preds: np.ndarray, targets: np.ndarray,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 20260804,
) -> Dict[str, float]:
    preds = np.asarray(preds); targets = np.asarray(targets)
    n = int(preds.size)
    if n < 2:
        return dict(rmse=float("nan"), rmse_ci95_lo=float("nan"),
                    rmse_ci95_hi=float("nan"))
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.sqrt(np.mean((preds[idx] - targets[idx]) ** 2))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return dict(rmse=rmse, rmse_ci95_lo=lo, rmse_ci95_hi=hi)


def auprc_with_permutation_p(
    preds: np.ndarray, targets: np.ndarray, positive_if_gt: float = 0.0,
    n_perm: int = 2000, seed: int = 20260804,
) -> Dict[str, float]:
    preds = np.asarray(preds); targets = np.asarray(targets)
    y = (targets > positive_if_gt).astype(int)
    n = int(y.size); npos = int(y.sum()); prev = float(npos / n) if n else float("nan")
    if n == 0 or npos == 0 or npos == n:
        return dict(auprc=float("nan"), auprc_pvalue=float("nan"),
                    n_positive=npos, prevalence=prev)
    observed = float(average_precision_score(y, preds))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if float(average_precision_score(y_perm, preds)) >= observed:
            hits += 1
    p = (hits + 1) / (n_perm + 1)
    return dict(auprc=observed, auprc_pvalue=float(p),
                n_positive=npos, prevalence=prev)


def per_fold_summary(fold_pccs: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([f for f in fold_pccs if np.isfinite(f)], dtype=float)
    if arr.size == 0:
        return dict(per_fold_pcc_mean=float("nan"),
                    per_fold_pcc_sd=float("nan"), per_fold_n=0)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return dict(per_fold_pcc_mean=mean, per_fold_pcc_sd=sd, per_fold_n=int(arr.size))



def steigers_z_dependent_pcc(
    preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray,
) -> Dict[str, float]:
    a = np.asarray(preds_a); b = np.asarray(preds_b); y = np.asarray(targets)
    n = int(y.size)
    if n < 4:
        return dict(delta_pcc=float("nan"), steiger_z=float("nan"),
                    steiger_pvalue=float("nan"), n=n)
    r_ay = stats.pearsonr(a, y)[0]
    r_by = stats.pearsonr(b, y)[0]
    r_ab = stats.pearsonr(a, b)[0]
    if not (np.isfinite(r_ay) and np.isfinite(r_by) and np.isfinite(r_ab)):
        return dict(delta_pcc=float("nan"), steiger_z=float("nan"),
                    steiger_pvalue=float("nan"), n=n)
    det = (1.0 - r_ay ** 2 - r_by ** 2 - r_ab ** 2 + 2.0 * r_ay * r_by * r_ab)
    if det <= 0 or n <= 3:
        return dict(delta_pcc=float(r_ay - r_by), steiger_z=float("nan"),
                    steiger_pvalue=float("nan"), n=n)
    t = (r_ay - r_by) * np.sqrt((n - 1) * (1.0 + r_ab) / (2.0 * ((n - 1) / (n - 3)) * det + ((r_ay + r_by) / 2.0) ** 2 * (1.0 - r_ab) ** 3))
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 3))
    return dict(delta_pcc=float(r_ay - r_by), steiger_z=float(t),
                steiger_pvalue=float(p), n=n)


def paired_bootstrap_delta(
    preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray,
    metric: str = "pcc", n_boot: int = 2000, seed: int = 20260804,
) -> Dict[str, float]:
    a = np.asarray(preds_a); b = np.asarray(preds_b); y = np.asarray(targets)
    n = int(y.size)
    rng = np.random.default_rng(seed)

    def _metric(p, t):
        if metric == "pcc":
            if np.std(p) == 0 or np.std(t) == 0:
                return float("nan")
            return float(stats.pearsonr(p, t)[0])
        elif metric == "rmse":
            return float(np.sqrt(np.mean((p - t) ** 2)))
        else:
            raise ValueError(metric)

    obs = _metric(a, y) - _metric(b, y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _metric(a[idx], y[idx]) - _metric(b[idx], y[idx])
    ci_lo = float(np.quantile(deltas, 0.025))
    ci_hi = float(np.quantile(deltas, 0.975))
    if obs >= 0:
        p_two = 2.0 * float(np.mean(deltas <= 0))
    else:
        p_two = 2.0 * float(np.mean(deltas >= 0))
    p_two = min(1.0, p_two)
    return dict(delta_obs=float(obs), delta_ci95_lo=ci_lo, delta_ci95_hi=ci_hi,
                bootstrap_pvalue=p_two, n_boot=n_boot, n=n)


def paired_t_on_abs_error(
    preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray,
) -> Dict[str, float]:
    a = np.abs(np.asarray(preds_a) - np.asarray(targets))
    b = np.abs(np.asarray(preds_b) - np.asarray(targets))
    t, p = stats.ttest_rel(a, b)
    return dict(paired_t=float(t), paired_t_pvalue=float(p),
                mean_abs_err_a=float(a.mean()), mean_abs_err_b=float(b.mean()),
                n=int(a.size))


def antisymmetry_bias_test(
    preds_fwd: np.ndarray, preds_inv: np.ndarray,
) -> Dict[str, float]:
    fwd = np.asarray(preds_fwd); inv = np.asarray(preds_inv)
    s = fwd + inv
    mean = float(s.mean())
    sd = float(s.std(ddof=1))
    n = int(s.size)
    if n < 2:
        return dict(antisym_bias_mean=mean, antisym_bias_sd=sd,
                    antisym_bias_pvalue=float("nan"), n=n)
    t, p = stats.ttest_1samp(s, 0.0)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    se = sd / np.sqrt(n)
    return dict(antisym_bias_mean=mean, antisym_bias_sd=sd,
                antisym_bias_ci95_lo=mean - tcrit * se,
                antisym_bias_ci95_hi=mean + tcrit * se,
                antisym_bias_t=float(t),
                antisym_bias_pvalue=float(p), n=n)



def full_benchmark_report(
    preds: np.ndarray, targets: np.ndarray,
    per_fold_pccs: Optional[Sequence[float]] = None,
    bootstrap_iters: int = 2000, permutation_iters: int = 2000,
) -> Dict[str, float]:
    out = {}
    out.update(pcc_with_stats(preds, targets))
    out.update(scc_with_stats(preds, targets))
    out.update(rmse_with_bootstrap_ci(preds, targets, n_boot=bootstrap_iters))
    out["mae"] = float(np.mean(np.abs(np.asarray(preds, dtype=float)
                                      - np.asarray(targets, dtype=float))))
    out.update(auprc_with_permutation_p(preds, targets, n_perm=permutation_iters))
    if per_fold_pccs is not None:
        out.update(per_fold_summary(per_fold_pccs))
    return out


def full_report(preds, targets, bootstrap_iters: int = 2000,
                permutation_iters: int = 2000) -> Dict[str, float]:
    return full_benchmark_report(-np.asarray(preds, dtype=float),
                                 -np.asarray(targets, dtype=float),
                                 bootstrap_iters=bootstrap_iters,
                                 permutation_iters=permutation_iters)
