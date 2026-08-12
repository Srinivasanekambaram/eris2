# ERIS2 — CPU image.
#
#   docker build -t eris2 .
#   docker run --rm -v "$PWD:/work" -w /work eris2 \
#       python scripts/predict.py --pdb data/example/pdbs/2ZTA.pdb \
#           --chain A --mutation E20Q K8A
#
# For GPU, see Dockerfile.gpu.

FROM python:3.8-slim-bookworm

# mkdssp computes solvent accessibility, secondary structure and backbone
# angles. Installing it from the distribution avoids the shared-library
# mismatches that occur when a prebuilt binary meets a different Boost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        dssp \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v mkdssp || ln -s "$(command -v dssp)" /usr/local/bin/mkdssp) \
    && mkdssp --version

WORKDIR /opt/eris2

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY eris2/     ./eris2/
COPY scripts/   ./scripts/
COPY configs/   ./configs/
COPY data/      ./data/
COPY model/     ./model/
COPY tests/     ./tests/
COPY README.md LICENSE ./

ENV PYTHONPATH=/opt/eris2 \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/hf

# ESM-2 (~2.5 GB) downloads on first use. Mount a volume here to keep it
# between runs:  -v eris2-cache:/cache
VOLUME ["/cache"]

# Fail the build if the package cannot be imported or DSSP is unusable.
RUN python -c "from eris2.data import require_dssp; require_dssp()" \
    && python -c "import eris2, torch, esm; print('ERIS2', eris2.__version__, '| torch', torch.__version__)"

ENTRYPOINT []
CMD ["python", "scripts/predict.py", "--help"]
