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
docker run --rm -v eris2-cache:/cache -v "$PWD:/work" -w /work "$IMAGE" \
    python /opt/eris2/scripts/batch_predict.py \
        --csv data/example/mutations.csv --pdb-dir data/example/pdbs \
        --cache-dir /work/cache --out /work/predictions.csv

docker run --rm -v "$PWD:/work" -w /work "$IMAGE" \
    python /opt/eris2/scripts/evaluate.py \
        --predictions /work/predictions.csv \
        --reference data/example/mutations.csv

echo
echo "Expected: PCC 0.7212  SCC 0.7471  RMSE 0.4671  MAE 0.3541"
