#!/bin/bash
set -e

export PYTHONPATH=$PWD/qmirt:$PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install --no-input -r requirements_sparse.txt

INPUT_DIR=$1
OUTPUT_DIR=$2
WORK_DIR="sparse_combine_work"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

mkdir -p "$OUTPUT_DIR"

echo "Combining sparse SRM chunks from $INPUT_DIR"

for tarball in "$INPUT_DIR"/*.tar.gz; do
    [ -e "$tarball" ] || continue
    tar -xzf "$tarball" -C "$WORK_DIR"
done

python3 payload/python/ospool_sparse_srm_combine.py \
    --input-dir "$WORK_DIR" \
    --output-dir "$OUTPUT_DIR"

tar -czf sparse_srm_final.tar.gz -C "$OUTPUT_DIR" .
