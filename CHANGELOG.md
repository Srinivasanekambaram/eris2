# Changelog

## v4 — 2026-08-12

Fixes found by running v3 from scratch on a clean Linux (aarch64) machine.
Predictions are unchanged: the example still gives PCC 0.7212, SCC 0.7471,
RMSE 0.4671, MAE 0.3541 on 19 mutations.

### Fixed — correctness

**Batch predictions could be attached to the wrong mutation.** `batch_predict.py`
built its output with `df.iloc[:len(ddg)]`, which assumes rows that fail to
featurise are at the end of the input. They are dropped wherever they occur, so
if a structure was missing, the surviving predictions were labelled with the
first N rows of the CSV instead. The numbers were correct and the
`(uniprot, mut)` labels were wrong — nothing looked broken, and `evaluate.py`
then scored mismatched pairs.

Each sample now carries its originating `row_index` through the model, and
`predict_batch` returns it. Callers join on it instead of slicing. The warning
also now names the dropped rows. Covered by `tests/test_row_alignment.py`.

**`scripts/train.py` could not be imported.** It referenced the pre-refactor
module names (`dataset_checking`, `model`, `mut_seq`) and was missing the
`sys.path` prelude every other script has. Now imports from `eris2.data`,
`eris2.model` and `eris2.training`.

**A malformed mutation aborted the whole run.** A value like `XYZ` raised
`ValueError: invalid literal for int()` from the feature builder. Malformed
mutations are now validated up front, warned about, and dropped like any other
unusable row.

### Fixed — portability

**DSSP failed on PDB files with no `HEADER` line.** mkdssp 4.2.x infers the
format from the first line, so a file starting with `ATOM` — normal for
structures stripped of HETATM records — is misread as mmCIF and produces no
output. Header-less inputs are now copied to a temporary file with a minimal
`HEADER` prepended before DSSP runs; coordinates are untouched and features are
identical. The bundled example structures also carry a `HEADER` now.

**Docker images did not build on aarch64.**
- CPU: the torch install used `--index-url`, which replaces PyPI entirely and
  left pip unable to fetch build backends; it is now `--extra-index-url`.
  `build-essential` added, since biopython compiles from source where no wheel
  exists.
- GPU: added `build-essential` and `python3.8-dev` — the CUDA base image has
  no compiler or Python headers.
- Both images now upgrade pip before installing.

### Fixed — documentation and packaging

- `conda install -c salilab dssp` replaced with `conda-forge`; the salilab
  channel is empty.
- The model section now states the expected file size and MD5 of
  `eris2_ensemble.pt`, and notes that the older `model.pth` /
  `model_quantum.pth` deposits are superseded and will not load.
- `pytest` added to `requirements.txt`.
- `torch.load` now uses `weights_only=True`, removing the FutureWarning.
- `verify_docker.sh` asserts the metrics rather than printing them, and uses a
  fresh cache directory so a container that cannot build features itself is not
  masked by host-computed cache entries.

### Known limitations

- GPU inference is unverified on aarch64: PyPI ships a CPU-only wheel for that
  architecture, so `torch.cuda.is_available()` is False there. Unaffected on
  x86_64.
- `scripts/train.py` has been exercised as a short two-fold smoke run, not a
  full training run, since v3.
