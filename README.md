# ERIS2

Predicts the change in protein stability (ΔΔG, kcal/mol) caused by a point
mutation, from the wild-type structure and the mutation. Positive ΔΔG means the
mutation is destabilising.

## Requirements

Tested on Linux with Python 3.8, PyTorch 2.4.1 (CUDA 12.1), mkdssp 4.4.10.
Exact package versions are in `requirements.txt`. A GPU is optional;
prediction runs on CPU.

### Docker (recommended)

```bash
docker build -t eris2 .                      # CPU
docker build -f Dockerfile.gpu -t eris2:gpu .   # CUDA 12.1

docker run --rm -v eris2-cache:/cache -v "$PWD:/work" -w /work eris2 \
    python /opt/eris2/scripts/predict.py \
        --pdb data/example/pdbs/2ZTA.pdb --chain A --mutation E20Q K8A
```

The image includes DSSP, so the shared-library problems that affect local
installs do not arise. `-v eris2-cache:/cache` keeps the ESM-2 download
(~2.5 GB) between runs. Add `--gpus all` with the GPU image.

If you built from a git clone the model weights are not included; mount them:

```bash
-v /path/to/eris2_ensemble.pt:/opt/eris2/model/eris2_ensemble.pt
```

### Conda

```bash
conda create -n eris2 python=3.8 -y
conda activate eris2
pip install -r requirements.txt
conda install -c conda-forge dssp -y   # provides mkdssp
```

DSSP is required. Check it before running anything:

```bash
mkdssp --version
```

If `mkdssp` is installed but fails to load its shared libraries, add the
environment's library directory to the loader path:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

ESM-2 (~2.5 GB) downloads automatically on first use. A GPU is optional;
prediction runs on CPU.

## Repository structure

```
model/        trained model and its configuration
data/         example inputs
eris2/        library code
scripts/      command-line tools
configs/      training configuration template
```

## Input data

A CSV with one row per mutation, and a directory of PDB files:

```
uniprot,chain,mut
2ZTA,A,E20Q
2ZTA,A,K8A
```

| column | meaning |
|---|---|
| `uniprot` | basename of the PDB file, so `2ZTA` means `2ZTA.pdb` in the PDB directory |
| `chain` | chain identifier |
| `mut` | wild-type residue, position, mutant residue — in **PDB numbering**, e.g. `E20Q` |
| `ddg` | optional; experimental value, used only for evaluation |

`data/example/` contains a working example: 19 mutations over 4 structures.

## Prediction

```bash
python scripts/predict.py \
    --pdb data/example/pdbs/2ZTA.pdb --chain A \
    --mutation E20Q K8A
```

```
  2ZTA A E20Q  ddG =  +0.20 kcal/mol   destabilising
  2ZTA A K8A   ddG =  +0.09 kcal/mol   destabilising
```

For many mutations:

```bash
python scripts/batch_predict.py \
    --csv data/example/mutations.csv \
    --pdb-dir data/example/pdbs \
    --out predictions.csv
```

`predictions.csv` contains `uniprot, chain, mut, ddg_pred` and one column per
ensemble member. If the input had a `ddg` column it is carried through as
`ddg_exp`, so evaluation can run without a separate reference file. Features are cached under `cache/`, so repeated runs on the
same structures are fast.

ERIS2 predicts with an ensemble of five models, shipped as a single file and
used automatically. `predict.py --show-folds` prints the individual members;
their spread indicates how consistent the ensemble is on that mutation.

## Evaluation

Scoring requires experimental values. Supply them in a CSV with a `ddg` column,
matched on `uniprot`, `chain`, `mut`:

```bash
python scripts/evaluate.py \
    --predictions predictions.csv \
    --reference data/example/mutations.csv
```

```
  n      19
  PCC    0.7212   95% CI [0.397, 0.885]   p = 4.93e-04
  SCC    0.7471   95% CI [0.443, 0.897]   p = 2.37e-04
  RMSE   0.4671   95% CI [0.286, 0.636]
  MAE    0.3541
  AUPRC  0.8572   stabilising mutations, baseline 0.421
```

Add `--plot scatter.png` for a predicted-against-experimental plot, or
`--out metrics.json` to save the numbers.

Running the example above reproduces these values exactly; a mismatch usually
means DSSP is not working.

## Model

`model/eris2_ensemble.pt` and `model/model_config.yaml` are found automatically.
To use a different checkpoint, pass `--checkpoint` and `--config`.

The weights are distributed as a release asset rather than in the git tree,
because of their size. Check that you have the right file:

| | |
|---|---|
| file | `model/eris2_ensemble.pt` |
| size | 112,054,558 bytes |
| MD5 | `6c8f7963e9ea3a89a7e2300d7dd16c0f` |
| contents | 5 cross-validation models, format `eris2-ensemble-v1` |

```bash
md5sum model/eris2_ensemble.pt
```

A prediction is the mean of the five models. No single fold is used, and no
checkpoint was selected on any benchmark. Earlier deposits contained files named
`model.pth` and `model_quantum.pth` from a previous, incompatible version of the
code; those are superseded and will not load here.

## Optional: precomputing features

Prediction builds its feature cache as it runs. For a large input, or to
separate a GPU step from a CPU step, precompute first:

```bash
python scripts/preprocess.py \
    --csv mutations.csv --pdb-dir pdbs/ --cache-dir cache
```

## Optional: training

Training uses cluster-aware cross-validation, so mutations in homologous
proteins never span training and test folds. This needs a `cluster_id` column,
which `scripts/cluster.py` adds using MMseqs2 (`conda install -c bioconda mmseqs2`):

```bash
python scripts/cluster.py \
    --csv train.csv --pdb-dir pdbs/ --out train_clustered.csv
```

Point `configs/train_example.yaml` at the clustered CSV and the PDB directory,
then:

```bash
python scripts/train.py --config configs/train_example.yaml
```

Checkpoints and per-fold metrics are written under the configured output
directory. `cluster_id` is used only for training; prediction does not need it.

## Complete example

```bash
python scripts/batch_predict.py \
    --csv data/example/mutations.csv \
    --pdb-dir data/example/pdbs \
    --out predictions.csv

python scripts/evaluate.py \
    --predictions predictions.csv \
    --reference data/example/mutations.csv
```

## Output files

| file | contents |
|---|---|
| `predictions.csv` | one row per mutation: `ddg_pred` plus each ensemble member |
| `metrics.json` | evaluation metrics, if `--out` was given |
| `cache/` | cached features; safe to delete, and rebuilt on demand |

## Citation

TBD (paper in revision).

## License

MIT — see `LICENSE`.
