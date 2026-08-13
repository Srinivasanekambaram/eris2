from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2 import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from eris2.data import DatasetConfig, ProteinDataset, require_dssp
from eris2.inference import collate, load_models, predict_batch

REQUIRED = ("uniprot", "chain", "mut")


def main():
    ap = argparse.ArgumentParser(description="Predict ΔΔG for a CSV of mutations")
    ap.add_argument("--csv", required=True,
                    help="Columns: uniprot, chain, mut (a ddg column is ignored here)")
    ap.add_argument("--pdb-dir", dest="pdb_dir", required=True,
                    help="Directory containing {uniprot}.pdb")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--cache-dir", dest="cache_dir", default="cache")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--device", default=None,
                    help="torch device, e.g. cuda, cuda:1 or cpu. Default: cuda when available, else cpu")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    args = ap.parse_args()

    require_dssp()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.csv}: missing required column(s) {missing}; "
                         f"expected {list(REQUIRED)}")
    if "ddg" not in df.columns:
        df = df.assign(ddg=0.0)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ProteinDataset(DatasetConfig(
        csv_file=df, pdb_folder=args.pdb_dir, cache_dir=args.cache_dir))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, collate_fn=collate)

    models = load_models(args.checkpoint, args.config, device)
    t0 = time.time()
    ddg, per_model, _, rows = predict_batch(models, loader, device)
    elapsed = time.time() - t0

    if len(ddg) == 0:
        raise SystemExit("No predictions produced. Check that the PDB files exist "
                         "and that each mutation's wild-type residue matches the structure.")

    if len(ddg) != len(df):
        dropped = df.index.difference(pd.Index(rows))
        print(f"[warn] {len(dropped)} of {len(df)} rows could not be featurised "
              f"and were dropped:")
        for i in dropped[:10]:
            r = df.loc[i]
            print(f"         row {i}: {r['uniprot']} {r['chain']} {r['mut']}")
        if len(dropped) > 10:
            print(f"         ... and {len(dropped) - 10} more")

    out = df.iloc[rows][list(REQUIRED)].copy()
    out["ddg_pred"] = ddg
    if "ddg" in df.columns and df["ddg"].abs().sum() > 0:
        out["ddg_exp"] = df["ddg"].iloc[rows].to_numpy()
    for i in range(per_model.shape[0]):
        out[f"fold_{i + 1}"] = per_model[i]
    out.to_csv(args.out, index=False)

    print(f"\n  {len(out)} predictions written to {args.out}  ({elapsed:.1f} s)")
    print(f"  ddG > 0 destabilising, ddG < 0 stabilising\n")


if __name__ == "__main__":
    main()
