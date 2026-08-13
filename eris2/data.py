from __future__ import annotations

import atexit
import gc
import hashlib
import math
import os
import pickle
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Bio import PDB
from Bio.PDB import DSSP, is_aa

CACHE_VERSION = "v2.0"

DSSP_HELP = """ERIS2 requires DSSP (`mkdssp`) for solvent accessibility, secondary
structure and backbone angles.

    conda install -c conda-forge dssp

If mkdssp is installed but fails to load its shared libraries, add the
environment's lib directory to the loader path:

    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
"""


def require_dssp() -> None:
    import shutil
    import subprocess

    exe = shutil.which("mkdssp") or shutil.which("dssp")
    detail = None
    if exe is None:
        detail = "mkdssp not found on PATH"
    else:
        try:
            r = subprocess.run([exe, "--version"], capture_output=True,
                               text=True, timeout=30)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip().splitlines()
                detail = f"{exe} fails to run: {err[0] if err else r.returncode}"
        except Exception as e:
            detail = f"{exe} could not be executed: {e}"
    if detail:
        raise SystemExit(f"\n{detail}\n\n{DSSP_HELP}")

ONEHOT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_ONEHOT_INDEX = {aa: i for i, aa in enumerate(ONEHOT_ALPHABET)}


@dataclass
class DatasetConfig:
    csv_file: str
    pdb_folder: str
    cache_dir: str = "cached_data"

    fallback_cache_dir: Optional[str] = None


    window_half: int = 15

    ca_radius_a: float = 15.0
    ca_neigh: int = 15
    atom_neigh: int = 100

    esm_model_name: str = "esm2_t33_650M_UR50D"

    hbond_distance_cutoff_a: float = 3.5

    use_disk_cache: bool = True

    device: Optional[str] = None
    
    offline_esm: bool = False

    use_onehot: bool = False

    zero_features: tuple = ()

    @property
    def window_len(self) -> int:
        return 2 * self.window_half + 1


DSSP_STATES = ("H", "G", "I", "E", "B", "T", "S", "-")
SS_MAPPING = {ss: [1 if i == j else 0 for j in range(8)] for i, ss in enumerate(DSSP_STATES)}

AA_CHARGE = {
    "R": 1, "K": 1, "H": 0.5,
    "D": -1, "E": -1,
    "A": 0, "N": 0, "C": 0, "G": 0, "I": 0,
    "L": 0, "M": 0, "F": 0, "P": 0, "S": 0,
    "T": 0, "W": 0, "Y": 0, "V": 0, "Q": 0,
}

AA_HYDROPHOBICITY = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

ATOM_TYPE_ONEHOT = {
    "C": [1, 0, 0, 0],
    "N": [0, 1, 0, 0],
    "O": [0, 0, 1, 0],
    "S": [0, 0, 0, 1],
}
ATOM_TYPE_OTHER = [0, 0, 0, 0]

SIDECHAIN_DONORS = {
    "ARG": ("NE", "NH1", "NH2"),
    "ASN": ("ND2",),
    "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"),
    "LYS": ("NZ",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1",),
    "TYR": ("OH",),
}
SIDECHAIN_ACCEPTORS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "ASN": ("OD1",),
    "GLN": ("OE1",),
    "HIS": ("ND1", "NE2"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
}


def get_sequence_and_mapping(
    structure: PDB.Structure.Structure, chain_id: str
) -> Tuple[str, Dict[int, int], Dict[int, int]]:
    aa3_to_aa1 = {aa3: aa1 for aa3, aa1 in zip(PDB.Polypeptide.aa3, PDB.Polypeptide.aa1)}
    seq_chars: List[str] = []
    pdb_to_seq: Dict[int, int] = {}
    seq_to_pdb: Dict[int, int] = {}

    for model in structure:
        if chain_id not in model:
            continue
        chain = model[chain_id]
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            pdb_num = residue.id[1]
            one_letter = aa3_to_aa1.get(residue.resname, "X")
            seq_idx = len(seq_chars)
            seq_chars.append(one_letter)
            pdb_to_seq[pdb_num] = seq_idx
            seq_to_pdb[seq_idx] = pdb_num
        return "".join(seq_chars), pdb_to_seq, seq_to_pdb

    return "", {}, {}


def get_ca_coordinates(structure: PDB.Structure.Structure, chain_id: str) -> np.ndarray:
    coords: List[np.ndarray] = []
    for model in structure:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if is_aa(residue, standard=True):
                if "CA" in residue:
                    coords.append(np.asarray(residue["CA"].coord, dtype=np.float32))
                else:
                    coords.append(np.full(3, np.nan, dtype=np.float32))
        break
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(coords, axis=0)


def get_all_atom_info(
    structure: PDB.Structure.Structure, chain_id: str
) -> Tuple[np.ndarray, List[str], List[int], List[Tuple[int, int]]]:
    atom_coords: List[np.ndarray] = []
    atom_elements: List[str] = []
    atom_residue_indices: List[int] = []
    residue_atom_ranges: List[Tuple[int, int]] = []

    for model in structure:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if not is_aa(residue, standard=True):
                continue
            start = len(atom_coords)
            res_idx = len(residue_atom_ranges)
            for atom in residue:
                atom_coords.append(np.asarray(atom.coord, dtype=np.float32))
                atom_elements.append(atom.element.strip().upper() if atom.element else "")
                atom_residue_indices.append(res_idx)
            end = len(atom_coords)
            residue_atom_ranges.append((start, end))
        break

    if not atom_coords:
        return np.zeros((0, 3), dtype=np.float32), [], [], []
    return (
        np.stack(atom_coords, axis=0),
        atom_elements,
        atom_residue_indices,
        residue_atom_ranges,
    )


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def get_ca_neighborhood(
    ca_coords: np.ndarray, mut_seq_idx: int, radius_a: float, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_res = ca_coords.shape[0]
    if n_res == 0 or mut_seq_idx < 0 or mut_seq_idx >= n_res:
        return np.zeros(0, dtype=np.int64), np.zeros((k, k), dtype=np.float32)

    dists_from_mut = np.linalg.norm(ca_coords - ca_coords[mut_seq_idx], axis=1)
    within = np.where(dists_from_mut <= radius_a)[0]
    within = within[np.argsort(dists_from_mut[within], kind="stable")]
    selected = within[:k]

    full_dist = compute_pairwise_distances(ca_coords[selected])
    submatrix = np.zeros((k, k), dtype=np.float32)
    n_sel = selected.shape[0]
    submatrix[:n_sel, :n_sel] = full_dist
    return selected.astype(np.int64), submatrix


def get_atom_neighborhood(
    atom_coords: np.ndarray,
    residue_atom_ranges: List[Tuple[int, int]],
    selected_residue_indices: np.ndarray,
    fixed_size: int,
) -> np.ndarray:
    if atom_coords.shape[0] == 0 or selected_residue_indices.shape[0] == 0:
        return np.zeros((fixed_size, fixed_size), dtype=np.float32)

    atom_indices: List[int] = []
    for res_idx in selected_residue_indices.tolist():
        start, end = residue_atom_ranges[res_idx]
        atom_indices.extend(range(start, end))
    atom_indices = atom_indices[:fixed_size]
    if not atom_indices:
        return np.zeros((fixed_size, fixed_size), dtype=np.float32)

    atom_arr = np.asarray(atom_indices, dtype=np.int64)
    sub_coords = atom_coords[atom_arr]
    full = compute_pairwise_distances(sub_coords)
    out = np.zeros((fixed_size, fixed_size), dtype=np.float32)
    n = full.shape[0]
    out[:n, :n] = full
    return out


def extract_window_1d(
    per_residue: np.ndarray, center: int, half: int, pad_value: float = 0.0
) -> np.ndarray:
    length = 2 * half + 1
    if per_residue.ndim == 1:
        out = np.full((length,), pad_value, dtype=per_residue.dtype)
    else:
        out = np.full((length, *per_residue.shape[1:]), pad_value, dtype=per_residue.dtype)

    src_start = max(0, center - half)
    src_end = min(per_residue.shape[0], center + half + 1)
    dst_start = src_start - (center - half)
    dst_end = dst_start + (src_end - src_start)
    if src_start < src_end:
        out[dst_start:dst_end] = per_residue[src_start:src_end]
    return out


def extract_window_mask(seq_length: int, center: int, half: int) -> np.ndarray:
    length = 2 * half + 1
    mask = np.zeros(length, dtype=np.float32)
    src_start = max(0, center - half)
    src_end = min(seq_length, center + half + 1)
    dst_start = src_start - (center - half)
    dst_end = dst_start + (src_end - src_start)
    if src_start < src_end:
        mask[dst_start:dst_end] = 1.0
    return mask


def get_physicochemical(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    charge = np.asarray([[AA_CHARGE.get(aa, 0.0)] for aa in sequence], dtype=np.float32)
    hydro = np.asarray([[AA_HYDROPHOBICITY.get(aa, 0.0)] for aa in sequence], dtype=np.float32)
    return charge, hydro


def get_atom_type_features(
    structure: PDB.Structure.Structure, chain_id: str
) -> np.ndarray:
    per_res: List[np.ndarray] = []
    for model in structure:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if not is_aa(residue, standard=True):
                continue
            atom_vecs = []
            for atom in residue:
                el = atom.element.strip().upper() if atom.element else ""
                atom_vecs.append(ATOM_TYPE_ONEHOT.get(el, ATOM_TYPE_OTHER))
            if atom_vecs:
                per_res.append(np.mean(atom_vecs, axis=0).astype(np.float32))
            else:
                per_res.append(np.zeros(4, dtype=np.float32))
        break
    if not per_res:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(per_res, dtype=np.float32)


MUTATION_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]-?\d+[ACDEFGHIKLMNPQRSTVWY]$")

_HEADER_FIXED: Dict[str, str] = {}


def _dssp_input(pdb_file: str) -> str:
    if pdb_file in _HEADER_FIXED:
        return _HEADER_FIXED[pdb_file]
    try:
        with open(pdb_file) as f:
            first = f.readline()
    except OSError:
        return pdb_file
    if first.startswith(("HEADER", "data_")):
        return pdb_file

    stem = Path(pdb_file).stem[:4].upper()
    header = f"HEADER    PROTEIN{' ' * 33}01-JAN-00   {stem:<4}\n"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    with open(pdb_file) as src:
        tmp.write(header)
        shutil.copyfileobj(src, tmp)
    tmp.close()
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.unlink(p))
    _HEADER_FIXED[pdb_file] = tmp.name
    return tmp.name


def calculate_dssp_features(
    pdb_file: str, structure: PDB.Structure.Structure, chain_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        dssp = DSSP(structure[0], _dssp_input(pdb_file), dssp="mkdssp")
    except Exception as e:

        raise RuntimeError(
            f"DSSP failed for {pdb_file}: {e}\n{DSSP_HELP}") from e

    rsa_out, ss_out, phi_psi_out = [], [], []
    for residue in structure[0][chain_id]:
        if not is_aa(residue, standard=True):
            continue
        key = (chain_id, residue.id)
        if key in dssp.keys():
            rsa_out.append(float(dssp[key][3]))
            ss_out.append(SS_MAPPING.get(dssp[key][2], SS_MAPPING["-"]))
            phi = float(dssp[key][4])
            psi = float(dssp[key][5])
            phi_val = phi / 180.0 if not math.isnan(phi) else 0.0
            psi_val = psi / 180.0 if not math.isnan(psi) else 0.0
            phi_psi_out.append([phi_val, psi_val])
        else:
            rsa_out.append(0.0)
            ss_out.append(SS_MAPPING["-"])
            phi_psi_out.append([0.0, 0.0])
            
    return (
        np.asarray(rsa_out, dtype=np.float32),
        np.asarray(ss_out, dtype=np.float32),
        np.asarray(phi_psi_out, dtype=np.float32),
    )


def calculate_hbonds(
    structure: PDB.Structure.Structure,
    chain_id: str,
    distance_cutoff: float = 3.5,
) -> np.ndarray:
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    features: List[List[float]] = []
    for model in structure:
        if chain_id not in model:
            continue
        residues = [r for r in model[chain_id] if is_aa(r, standard=True)]

        for res in residues:
            bb = 0
            sc = 0

            if "N" in res:
                n_coord = np.asarray(res["N"].coord)
                for other in residues:
                    if other is res or "O" not in other:
                        continue
                    o_coord = np.asarray(other["O"].coord)
                    if _dist(n_coord, o_coord) <= distance_cutoff:
                        bb += 1

            donor_set = SIDECHAIN_DONORS.get(res.get_resname(), ())
            for donor_atom in donor_set:
                if donor_atom not in res:
                    continue
                d_coord = np.asarray(res[donor_atom].coord)
                for other in residues:
                    if other is res:
                        continue
                    acc_set = SIDECHAIN_ACCEPTORS.get(other.get_resname(), ())
                    for acc_atom in acc_set:
                        if acc_atom not in other:
                            continue
                        if _dist(d_coord, np.asarray(other[acc_atom].coord)) <= distance_cutoff:
                            sc += 1

            total = bb + sc
            features.append([
                min(bb / 4.0, 1.0),
                min(sc / 6.0, 1.0),
                min(total / 10.0, 1.0),
            ])
        break

    if not features:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(features, dtype=np.float32)


def _pad_to_length(arr: np.ndarray, target_len: int, width: Optional[int] = None) -> np.ndarray:
    if arr.shape[0] == 0:
        if width is None:
            return np.zeros(target_len, dtype=np.float32)
        return np.zeros((target_len, width), dtype=np.float32)
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    if arr.ndim == 1:
        out = np.zeros(target_len, dtype=arr.dtype)
        out[: arr.shape[0]] = arr
        return out
    out = np.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


class ProteinDataset(Dataset):

    def __init__(self, config: DatasetConfig):
        self.config = config
        if isinstance(config.csv_file, pd.DataFrame):
            self.data = config.csv_file
        else:
            self.data = pd.read_csv(config.csv_file)

        self.pdb_folder = Path(config.pdb_folder)
        self.cache_dir = Path(config.cache_dir)
        self.fallback_cache_dir = (
            Path(config.fallback_cache_dir) if config.fallback_cache_dir else None)
        if config.use_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.fallback_cache_dir is not None:
            if not self.fallback_cache_dir.exists():
                raise FileNotFoundError(
                    f"fallback_cache_dir does not exist: {self.fallback_cache_dir}")
            if self.fallback_cache_dir.resolve() == self.cache_dir.resolve():
                raise ValueError(
                    "fallback_cache_dir must differ from cache_dir; pointing them "
                    "at the same directory would make the read-only guarantee "
                    "meaningless.")

        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._esm_model = None
        self._batch_converter = None


    def _ensure_esm_loaded(self) -> None:
        if self._esm_model is not None:
            return
        if self.config.offline_esm:
            raise RuntimeError("offline_esm is True but ESM model was requested. Ensure embeddings are precomputed.")
        import esm
        model_fn = getattr(esm.pretrained, self.config.esm_model_name)
        model, alphabet = model_fn()
        model.eval()
        self._esm_model = model.to(self.device)
        self._batch_converter = alphabet.get_batch_converter()

    def get_embedding(self, sequence: str) -> torch.Tensor:
        if self.config.use_disk_cache:
            cache_key = hashlib.md5(f"{CACHE_VERSION}|{sequence}".encode()).hexdigest()
            emb_path = self.cache_dir / f"embedding_{cache_key}.pt"
            if emb_path.exists():
                return torch.load(emb_path, weights_only=True)

        self._ensure_esm_loaded()
        with torch.no_grad():
            _, _, batch_tokens = self._batch_converter([(None, sequence)])
            batch_tokens = batch_tokens.to(self.device)
            results = self._esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            full = results["representations"][33].squeeze(0)
            embeddings = full[1 : 1 + len(sequence)].detach().cpu()

        if self.config.use_disk_cache:
            torch.save(embeddings, emb_path)
        return embeddings

    def _load_structure(self, uniprot_id: str) -> Optional[PDB.Structure.Structure]:
        pdb_path = self.pdb_folder / f"{uniprot_id}.pdb"
        if not pdb_path.exists():
            return None
        return self.pdb_parser.get_structure(uniprot_id, str(pdb_path))

    def _apply_zeroing(self, sample):
        for key in getattr(self.config, "zero_features", ()) or ():
            if key not in sample:
                raise KeyError(
                    f"zero_features lists {key!r}, which is not a sample tensor. "
                    f"Available: {sorted(k for k in sample if k != 'ddg')}")
            sample[key] = torch.zeros_like(sample[key])
        return sample

    def _cache_name(self, uniprot_id: str, chain_id: str, mutation: str) -> str:
        variant = "|onehot" if getattr(self.config, "use_onehot", False) else ""
        key = f"{CACHE_VERSION}{variant}|{uniprot_id}|{chain_id}|{mutation}"
        return f"{hashlib.md5(key.encode()).hexdigest()}.pkl"

    def _cache_path(self, uniprot_id: str, chain_id: str, mutation: str) -> Path:
        return self.cache_dir / self._cache_name(uniprot_id, chain_id, mutation)

    def _fallback_path(self, uniprot_id: str, chain_id: str, mutation: str):
        if self.fallback_cache_dir is None:
            return None
        return self.fallback_cache_dir / self._cache_name(
            uniprot_id, chain_id, mutation)

    def build_sample(
        self, uniprot_id: str, chain_id: str, mutation: str, ddg: float
    ) -> Optional[dict]:
        structure = self._load_structure(uniprot_id)
        if structure is None:
            return None

        pdb_file = str(self.pdb_folder / f"{uniprot_id}.pdb")
        sequence, pdb_to_seq, _seq_to_pdb = get_sequence_and_mapping(structure, chain_id)
        if not sequence:
            return None

        if not MUTATION_RE.match(str(mutation)):
            warnings.warn(
                f"skipping malformed mutation {mutation!r} for {uniprot_id} "
                f"chain {chain_id}: expected e.g. C191F (wild-type residue, "
                f"position in PDB numbering, mutant residue)")
            return None

        wt_aa = mutation[0]
        pdb_position = int(mutation[1:-1])
        mut_aa = mutation[-1]
        if pdb_position not in pdb_to_seq:
            return None
        seq_position = pdb_to_seq[pdb_position]
        if sequence[seq_position] != wt_aa:
            return None

        w = self.config.window_half
        mutated_sequence = sequence[:seq_position] + mut_aa + sequence[seq_position + 1 :]

        if self.config.use_onehot:
            def _onehot(seq: str) -> np.ndarray:
                arr = np.zeros((len(seq), 20), dtype=np.float32)
                for i, aa in enumerate(seq):
                    j = _ONEHOT_INDEX.get(aa)
                    if j is not None:
                        arr[i, j] = 1.0
                return arr
            wt_full_np = _onehot(sequence)
            mut_full_np = _onehot(mutated_sequence)
            wt_window = extract_window_1d(wt_full_np, seq_position, w, pad_value=0.0).astype(np.float32)
            mut_window = extract_window_1d(mut_full_np, seq_position, w, pad_value=0.0).astype(np.float32)
        else:

            wt_full = self.get_embedding(sequence)
            mut_full = self.get_embedding(mutated_sequence)

            wt_window = extract_window_1d(wt_full.numpy(), seq_position, w).astype(np.float32)
            mut_window = extract_window_1d(mut_full.numpy(), seq_position, w).astype(np.float32)
        window_mask = extract_window_mask(len(sequence), seq_position, w)

        ca_coords = get_ca_coordinates(structure, chain_id)
        selected_res, ca_submatrix = get_ca_neighborhood(
            ca_coords, seq_position, self.config.ca_radius_a, self.config.ca_neigh
        )
        atom_coords, _elements, _atom_res, res_atom_ranges = get_all_atom_info(
            structure, chain_id
        )
        atom_submatrix = get_atom_neighborhood(
            atom_coords, res_atom_ranges, selected_res, self.config.atom_neigh
        )

        rsa_full, ss_full, phi_psi_full = calculate_dssp_features(pdb_file, structure, chain_id)
        hbond_full = calculate_hbonds(
            structure,
            chain_id,
            distance_cutoff=self.config.hbond_distance_cutoff_a,
        )
        charge_full, hydro_full = get_physicochemical(sequence)
        atom_type_full = get_atom_type_features(structure, chain_id)

        rsa_full = _pad_to_length(rsa_full, len(sequence))
        ss_full = _pad_to_length(ss_full, len(sequence), width=8)
        phi_psi_full = _pad_to_length(phi_psi_full, len(sequence), width=2)
        hbond_full = _pad_to_length(hbond_full, len(sequence), width=3)
        atom_type_full = _pad_to_length(atom_type_full, len(sequence), width=4)

        rsa_w = extract_window_1d(rsa_full, seq_position, w)
        angles_w = extract_window_1d(phi_psi_full, seq_position, w)
        hbond_w = extract_window_1d(hbond_full, seq_position, w)
        ss_w = extract_window_1d(ss_full, seq_position, w)
        charge_w = extract_window_1d(charge_full, seq_position, w)
        hydro_w = extract_window_1d(hydro_full, seq_position, w)
        atom_type_w = extract_window_1d(atom_type_full, seq_position, w)

        return {
            "wt_embedding": torch.tensor(wt_window, dtype=torch.float32),
            "mut_embedding": torch.tensor(mut_window, dtype=torch.float32),
            "ca_distance_matrix": torch.tensor(ca_submatrix, dtype=torch.float32),
            "atom_distance_matrix": torch.tensor(atom_submatrix, dtype=torch.float32),
            "rsa_values": torch.tensor(rsa_w, dtype=torch.float32),
            "backbone_angles": torch.tensor(angles_w, dtype=torch.float32),
            "hbond_features": torch.tensor(hbond_w, dtype=torch.float32),
            "ss_features": torch.tensor(ss_w, dtype=torch.float32),
            "charge_features": torch.tensor(charge_w, dtype=torch.float32),
            "hydrophobicity_features": torch.tensor(hydro_w, dtype=torch.float32),
            "atom_features": torch.tensor(atom_type_w, dtype=torch.float32),
            "window_mask": torch.tensor(window_mask, dtype=torch.float32),
            "ddg": torch.tensor(float(ddg), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[dict]:
        row = self.data.iloc[idx]
        uniprot_id = row["uniprot"]
        chain_id = row["chain"]
        mutation = row["mut"]
        ddg = row["ddg"]

        if self.config.use_disk_cache:
            cache_path = self._cache_path(uniprot_id, chain_id, mutation)
            fallback_path = self._fallback_path(uniprot_id, chain_id, mutation)
            for path, writable in ((cache_path, True), (fallback_path, False)):
                if path is None or not path.exists():
                    continue
                try:
                    with open(path, "rb") as f:
                        sample = pickle.load(f)
                except Exception:
                    if writable:
                        path.unlink(missing_ok=True)
                    continue
                sample["ddg"] = torch.tensor(float(ddg), dtype=torch.float32)
                sample["row_index"] = torch.tensor(int(idx), dtype=torch.long)
                return self._apply_zeroing(sample)

        try:
            sample = self.build_sample(uniprot_id, chain_id, mutation, ddg)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if sample is None:
            return None

        if self.config.use_disk_cache:
            with open(self._cache_path(uniprot_id, chain_id, mutation), "wb") as f:
                pickle.dump(sample, f)
        sample["row_index"] = torch.tensor(int(idx), dtype=torch.long)
        return self._apply_zeroing(sample)
