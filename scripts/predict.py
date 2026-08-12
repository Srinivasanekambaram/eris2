from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2 import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from eris2.data import DatasetConfig, ProteinDataset, require_dssp
from eris2.inference import collate, load_models, predict_batch

MUTATION = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]$")


def predict(pdb, chain, mutations, checkpoint, config, cache_dir, device=None):
    bad = [m for m in mutations if not MUTATION.match(m)]
    if bad:
        raise ValueError(f"malformed mutation(s) {bad}; expected e.g. C191F "
                         f"(wild-type residue, position in PDB numbering, mutant residue)")

    pdb = Path(pdb).resolve()
    if not pdb.exists():
        raise FileNotFoundError(pdb)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.DataFrame({"uniprot": pdb.stem, "chain": chain,
                          "mut": mutations, "ddg": 0.0})
    dataset = ProteinDataset(DatasetConfig(
        csv_file=frame, pdb_folder=str(pdb.parent), cache_dir=cache_dir))
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(mutations),
                                         shuffle=False, collate_fn=collate)

    models = load_models(checkpoint, config, device)
    ddg, per_model, _ = predict_batch(models, loader, device)

    if len(ddg) != len(mutations):
        raise RuntimeError(
            f"{len(mutations) - len(ddg)} of {len(mutations)} mutations could not "
            f"be featurised from {pdb.name} chain {chain}. Check that the position "
            f"exists in the structure and that the wild-type residue matches.")

    out = pd.DataFrame({"pdb": pdb.stem, "chain": chain, "mutation": mutations,
                        "ddg_pred": ddg})
    for i in range(per_model.shape[0]):
        out[f"fold_{i + 1}"] = per_model[i]
    out["fold_sd"] = per_model.std(axis=0, ddof=1) if per_model.shape[0] > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Predict ΔΔG for a single structure")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--mutation", required=True, nargs="+",
                    help="One or more mutations, e.g. C191F (PDB numbering)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--cache-dir", dest="cache_dir", default="cache")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    ap.add_argument("--show-folds", dest="show_folds", action="store_true",
                    help="Print each ensemble member's prediction")
    args = ap.parse_args()

    require_dssp()
    df = predict(args.pdb, args.chain, args.mutation,
                 args.checkpoint, args.config, args.cache_dir)

    width = max(len(m) for m in df["mutation"])
    print()
    for _, r in df.iterrows():
        effect = "destabilising" if r["ddg_pred"] > 0 else "stabilising"
        print(f"  {r['pdb']} {r['chain']} {r['mutation']:<{width}}  "
              f"ddG = {r['ddg_pred']:+6.2f} kcal/mol   {effect}")
        if args.show_folds:
            folds = "  ".join(f"{r[c]:+.2f}" for c in df.columns
                              if c.startswith("fold_") and c != "fold_sd")
            print(f"  {'':<{width + 10}}folds: {folds}   SD {r['fold_sd']:.2f}")
    print()

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"  written to {args.out}\n")


if __name__ == "__main__":
    main()
