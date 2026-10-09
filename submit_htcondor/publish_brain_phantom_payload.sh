#!/bin/bash
# Build the brain SPECT phantom job payload and publish it to OSDF (same scheme as
# publish_payload_osdf.sh: content-hashed name, so a cached copy can never be stale).
#
# Contents: payload/python, persistent_data/{brain_spect/csv,GateMaterials.db}, qmirt/src,
# payload/phantom_specs/<name>.json (one per --spec NAME=SPEC.json) and the image phantom
# data the specs need (payload/phantom_sources/<model dir>/: GATE meshes, labels, activity).
# The CSG shield needs no STL.
#
#   publish_brain_phantom_payload.sh --spec small_jaszczak=PATH/spec.json --spec mesh50_head=PATH/spec.json \
#       [--dest ospool:/ospool/ap40/data/fang.han/payload | --dest DIR] [--keep-local DIR]
# Prints the osdf:// URL on the last line.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${BRAIN_PAYLOAD_DEST:-ospool:/ospool/ap40/data/fang.han/payload}"
OSDF_PREFIX="${OSDF_PREFIX:-osdf:///ospool/ap40/data/fang.han/payload}"
PHANTOM_DATA="${QMIRT_PHANTOM_DATA:-$REPO_ROOT/dev/python/phantom_sources}"
KEEP_LOCAL=""
SPECS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec) SPECS+=("$2"); shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --keep-local) KEEP_LOCAL="$2"; shift 2 ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; exit 2 ;;
    esac
done
[[ ${#SPECS[@]} -gt 0 ]] || { echo "Give at least one --spec NAME=SPEC.json" >&2; exit 2; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE" "${STAGE}.tar.gz"' EXIT
mkdir -p "$STAGE/payload/python" "$STAGE/payload/phantom_specs" "$STAGE/persistent_data/brain_spect" "$STAGE/qmirt"
cp "$REPO_ROOT"/payload/python/*.py "$STAGE/payload/python/"
cp -r "$REPO_ROOT/persistent_data/brain_spect/csv" "$STAGE/persistent_data/brain_spect/"
cp "$REPO_ROOT/persistent_data/GateMaterials.db" "$STAGE/persistent_data/"
cp -r "$REPO_ROOT/qmirt/src" "$STAGE/qmirt/"

for item in "${SPECS[@]}"; do
    name="${item%%=*}"; spec="${item#*=}"
    # only the phantom definition (spec.json of a bundle also carries the bundle record)
    python3 - "$spec" "$STAGE/payload/phantom_specs/${name}.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump({k: d[k] for k in ("phantom", "pose", "options", "model_version") if k in d}, open(sys.argv[2], "w"), indent=1)
PY
    # image phantoms: their data directory (model -> directory_name, files)
    dir_and_files="$(cd "$REPO_ROOT/payload/python" && python3 - "$STAGE/payload/phantom_specs/${name}.json" <<'PY'
import json, sys
import phantom_models as pm
d = json.load(open(sys.argv[1]))
if pm.is_image(d["phantom"]):
    m = pm.IMAGE_MODELS[d["phantom"]]
    files = {m.activity, m.attenuation, m.activity.replace(".mhd", ".raw"), m.attenuation.replace(".mhd", ".raw")}
    files |= {r["file"] for r in (m.meshes or [])}
    print(m.directory_name, " ".join(sorted(files)))
PY
)"
    if [[ -n "$dir_and_files" ]]; then
        read -r model_dir files <<<"$dir_and_files"
        mkdir -p "$STAGE/payload/phantom_sources/$model_dir"
        for f in $files; do cp "$PHANTOM_DATA/$model_dir/$f" "$STAGE/payload/phantom_sources/$model_dir/"; done
        [[ -f "$PHANTOM_DATA/$model_dir/README.md" ]] && cp "$PHANTOM_DATA/$model_dir/README.md" "$PHANTOM_DATA/$model_dir/LICENSE" "$STAGE/payload/phantom_sources/$model_dir/" 2>/dev/null || true
    fi
done

find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.egg-info' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -type f -delete
HASH="$(cd "$STAGE" && find . -type f -exec sha256sum {} + | LC_ALL=C sort -k2 | sha256sum | cut -c1-12)"
NAME="qmirt-brain-phantom-payload-${HASH}.tar.gz"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2020-01-01' \
    -C "$STAGE" -cf - payload persistent_data qmirt | gzip -n -9 > "${STAGE}.tar.gz"
echo "Payload $NAME: $(du -h "${STAGE}.tar.gz" | cut -f1); specs: $(ls "$STAGE/payload/phantom_specs" | tr '\n' ' ')" >&2

if [[ -n "$KEEP_LOCAL" ]]; then mkdir -p "$KEEP_LOCAL"; cp "${STAGE}.tar.gz" "$KEEP_LOCAL/$NAME"; fi
if [[ "$DEST" == *:* ]]; then  # remote (host:dir): copy then rename, so a job never sees a partial file
    host="${DEST%%:*}"; dir="${DEST#*:}"
    if ssh -o BatchMode=yes "$host" "test -f '$dir/$NAME'"; then
        echo "Already published: $DEST/$NAME" >&2
    else
        scp -q "${STAGE}.tar.gz" "$host:$dir/$NAME.tmp" && ssh -o BatchMode=yes "$host" "mv '$dir/$NAME.tmp' '$dir/$NAME'"
        echo "Published: $DEST/$NAME" >&2
    fi
else
    mkdir -p "$DEST"; [[ -f "$DEST/$NAME" ]] || { cp "${STAGE}.tar.gz" "$DEST/$NAME.tmp"; mv "$DEST/$NAME.tmp" "$DEST/$NAME"; }
fi
echo "${OSDF_PREFIX}/${NAME}"
