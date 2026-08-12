from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from Bio import PDB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eris2.data import get_sequence_and_mapping


def write_fasta(df: pd.DataFrame, pdb_dir: Path, path: Path) -> int:
    parser = PDB.PDBParser(QUIET=True)
    n = 0
    with open(path, "w") as fh:
        for uniprot, chain in df[["uniprot", "chain"]].drop_duplicates().itertuples(index=False):
            pdb = pdb_dir / f"{uniprot}.pdb"
            if not pdb.exists():
                print(f"  [skip] {pdb.name} not found")
                continue
            try:
                seq, _, _ = get_sequence_and_mapping(
                    parser.get_structure(uniprot, str(pdb)), chain)
            except Exception as e:
                print(f"  [skip] {uniprot} {chain}: {e}")
                continue
            if seq:
                fh.write(f">{uniprot}_{chain}\n{seq}\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Assign cluster_id by MMseqs2 clustering")
    ap.add_argument("--csv", required=True, help="Columns: uniprot, chain, mut, ddg")
    ap.add_argument("--pdb-dir", dest="pdb_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-seq-id", dest="min_seq_id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        raise SystemExit("mmseqs not found on PATH. "
                         "Install with: conda install -c bioconda mmseqs2")

    df = pd.read_csv(args.csv)
    for col in ("uniprot", "chain"):
        if col not in df.columns:
            raise SystemExit(f"{args.csv}: missing required column '{col}'")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta = tmp / "sequences.fasta"
        n = write_fasta(df, Path(args.pdb_dir), fasta)
        if n == 0:
            raise SystemExit("no sequences could be extracted from the PDB files")
        print(f"  {n} sequences extracted")

        prefix = tmp / "cluster_result"
        subprocess.run([mmseqs, "easy-cluster", str(fasta), str(prefix), str(tmp / "tmp"),
                        "--min-seq-id", str(args.min_seq_id),
                        "-c", str(args.coverage), "--cov-mode", "1",
                        "--threads", str(args.threads)], check=True)

        clusters = pd.read_csv(f"{prefix}_cluster.tsv", sep="\t",
                               names=["cluster_id", "member"])

    ids = clusters["member"].str.rsplit("_", n=1, expand=True)
    clusters["uniprot"], clusters["chain"] = ids[0], ids[1]
    lookup = clusters.set_index(["uniprot", "chain"])["cluster_id"]

    df["cluster_id"] = list(
        lookup.reindex(list(zip(df["uniprot"].astype(str), df["chain"].astype(str)))))
    unassigned = int(df["cluster_id"].isna().sum())
    if unassigned:
        print(f"  [warn] {unassigned} rows had no sequence and were dropped")
        df = df.dropna(subset=["cluster_id"])

    df.to_csv(args.out, index=False)
    print(f"\n  {len(df)} mutations in {df['cluster_id'].nunique()} clusters "
          f"-> {args.out}\n")


if __name__ == "__main__":
    main()
