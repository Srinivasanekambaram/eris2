#!/usr/bin/env bash
# Build the image and confirm it reproduces the published example.
set -euo pipefail
IMAGE=${1:-eris2}
FILE=${2:-Dockerfile}

echo "building $IMAGE from $FILE ..."
docker build -f "$FILE" -t "$IMAGE" .

echo
echo "checking DSSP inside the image ..."
docker run --rm "$IMAGE" mkdssp --version

echo
echo "running the example (first run downloads ESM-2, ~2.5 GB) ..."
# A fresh cache directory each time: reusing a host-populated cache would hide
# a container that cannot build features itself, which is exactly the failure
# this script exists to catch.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

docker run --rm -v eris2-cache:/cache -v "$PWD:/work" -v "$WORK:/scratch" -w /work "$IMAGE" \
    python /opt/eris2/scripts/batch_predict.py \
        --csv data/example/mutations.csv --pdb-dir data/example/pdbs \
        --cache-dir /scratch/cache --out /scratch/predictions.csv

docker run --rm -v "$PWD:/work" -v "$WORK:/scratch" -w /work "$IMAGE" \
    python /opt/eris2/scripts/evaluate.py \
        --predictions /scratch/predictions.csv \
        --reference data/example/mutations.csv \
        --out /scratch/metrics.json

echo
python3 - "$WORK/metrics.json" <<'PY'
import json, sys

expected = {"pcc": 0.7212, "scc": 0.7471, "rmse": 0.4671, "mae": 0.3541}
got = json.load(open(sys.argv[1]))
bad = []
for k, want in expected.items():
    have = got.get(k)
    if have is None or abs(have - want) > 5e-4:
        bad.append(f"  {k.upper():5s} expected {want:.4f}, got "
                   f"{'missing' if have is None else format(have, '.4f')}")
    else:
        print(f"  {k.upper():5s} {have:.4f}  ok")
if bad:
    print("\nFAILED — the container does not reproduce the published example:")
    print("\n".join(bad))
    sys.exit(1)
print("\nOK — container reproduces the published example.")
PY
