from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2.metrics import full_report

KEY = ["uniprot", "chain", "mut"]


def main():
    ap = argparse.ArgumentParser(description="Evaluate ΔΔG predictions")
    ap.add_argument("--predictions", required=True,
                    help="Output of batch_predict.py (needs ddg_pred)")
    ap.add_argument("--reference", default=None,
                    help="CSV with a ddg column. Omit if --predictions already "
                         "carries ddg_exp.")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--plot", default=None, help="Optional scatter plot path (.png)")
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    if "ddg_pred" not in pred.columns:
        raise SystemExit(f"{args.predictions}: no ddg_pred column")

    if args.reference:
        ref = pd.read_csv(args.reference)
        if "ddg" not in ref.columns:
            raise SystemExit(f"{args.reference}: no ddg column to evaluate against")
        merged = (pred.drop(columns=["ddg_exp"], errors="ignore")
                      .merge(ref[KEY + ["ddg"]], on=KEY, how="inner")
                      .rename(columns={"ddg": "ddg_exp"}))
    elif "ddg_exp" in pred.columns:
        merged = pred
    else:
        raise SystemExit("Provide --reference, or a predictions file containing ddg_exp")

    merged = merged.dropna(subset=["ddg_pred", "ddg_exp"])
    if len(merged) < 3:
        raise SystemExit(f"only {len(merged)} rows matched between the two files; "
                         f"check that uniprot/chain/mut agree")
    if len(merged) < len(pred):
        print(f"[warn] {len(pred) - len(merged)} prediction rows had no reference value")

    p = merged["ddg_pred"].to_numpy(float)
    t = merged["ddg_exp"].to_numpy(float)
    m = full_report(p, t)

    print(f"\n  n      {m['n']}")
    print(f"  PCC    {m['pcc']:.4f}   95% CI [{m['pcc_ci95_lo']:.3f}, {m['pcc_ci95_hi']:.3f}]   p = {m['pcc_pvalue']:.2e}")
    print(f"  SCC    {m['scc']:.4f}   95% CI [{m['scc_ci95_lo']:.3f}, {m['scc_ci95_hi']:.3f}]   p = {m['scc_pvalue']:.2e}")
    print(f"  RMSE   {m['rmse']:.4f}   95% CI [{m['rmse_ci95_lo']:.3f}, {m['rmse_ci95_hi']:.3f}]")
    print(f"  MAE    {m['mae']:.4f}")
    print(f"  AUPRC  {m['auprc']:.4f}   stabilising mutations, baseline {m['prevalence']:.3f}\n")

    if m["pcc"] < 0:
        print("  [warn] negative correlation. If your reference values use the "
              "opposite sign (negative = destabilising), flip them before "
              "evaluating.\n")

    if args.out:
        json.dump(m, open(args.out, "w"), indent=2)
        print(f"  metrics written to {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(t, p, s=14, alpha=0.7, edgecolor="none")
        lo, hi = min(t.min(), p.min()) - 0.5, max(t.max(), p.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("experimental ddG (kcal/mol)")
        ax.set_ylabel("predicted ddG (kcal/mol)")
        ax.set_title(f"n = {m['n']}   r = {m['pcc']:.2f}")
        fig.tight_layout(); fig.savefig(args.plot, dpi=200)
        print(f"  plot written to {args.plot}")
    print()


if __name__ == "__main__":
    main()
