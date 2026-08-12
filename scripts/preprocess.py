from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2.data import DatasetConfig, ProteinDataset, require_dssp

REQUIRED = ("uniprot", "chain", "mut")


def main():
    ap = argparse.ArgumentParser(description="Precompute ERIS2 input features")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pdb-dir", dest="pdb_dir", required=True)
    ap.add_argument("--cache-dir", dest="cache_dir", default="cache")
    args = ap.parse_args()

    require_dssp()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.csv}: missing required column(s) {missing}")
    if "ddg" not in df.columns:
        df = df.assign(ddg=0.0)

    dataset = ProteinDataset(DatasetConfig(
        csv_file=df, pdb_folder=args.pdb_dir, cache_dir=args.cache_dir))

    t0 = time.time()
    ok = failed = 0
    for i in range(len(dataset)):
        if dataset[i] is None:
            failed += 1
            r = df.iloc[i]
            print(f"  [skip] {r['uniprot']} {r['chain']} {r['mut']}")
        else:
            ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dataset)}")

    print(f"\n  cached {ok}/{len(dataset)} mutations in {time.time() - t0:.1f} s"
          f" -> {args.cache_dir}")
    if failed:
        print(f"  {failed} could not be featurised: the position is absent from "
              f"the structure, or the wild-type residue does not match.")


if __name__ == "__main__":
    main()
