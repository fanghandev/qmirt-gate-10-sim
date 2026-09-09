#!/bin/bash
# Build and publish the cardiac job payload to OSDF.
#
# Jobs pull this tarball through the site cache (like the .sif) instead of having
# the access point re-send the payload for every job. The filename embeds a content
# hash: OSDF caches key on path, so overwriting a fixed name can leave execute nodes
# reading a stale payload for hours. A new content hash is a new path, which can
# never be stale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# On the access point this is a real path; on the workstation it is the sshfs mount.
if [[ -d /ospool/ap40/data/fang.han ]]; then
    DEFAULT_DEST="/ospool/ap40/data/fang.han/payload"
else
    DEFAULT_DEST="${OSPOOL_MOUNT:-$HOME/ospool}/payload"
fi
DEST_DIR="${OSDF_DEST_DIR:-$DEFAULT_DEST}"
OSDF_PREFIX="${OSDF_PREFIX:-osdf:///ospool/ap40/data/fang.han/payload}"
FORCE=0
QUIET=0

usage() {
    echo "Usage: $0 [--dest-dir DIR] [--osdf-prefix URL] [--force] [--quiet]"
    echo ""
    echo "Builds the minimal cardiac payload (payload/python, persistent_data/cardiac_spect,"
    echo "GateMaterials.db, qmirt/src), names it by content hash, and publishes it to the"
    echo "OSDF-backed data area. Prints the osdf:// URL on the last line."
    echo ""
    echo "Re-running with unchanged sources is a no-op and prints the existing URL."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest-dir) DEST_DIR="$2"; shift 2 ;;
        --osdf-prefix) OSDF_PREFIX="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --quiet) QUIET=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unexpected argument: $1" >&2; usage; exit 2 ;;
    esac
done

log() { [[ "$QUIET" -eq 1 ]] || echo "$@" >&2; }

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

mkdir -p "$STAGE_DIR/payload/python" "$STAGE_DIR/persistent_data" "$STAGE_DIR/qmirt"
cp "$REPO_ROOT"/payload/python/*.py "$STAGE_DIR/payload/python/"
cp -r "$REPO_ROOT/persistent_data/cardiac_spect" "$STAGE_DIR/persistent_data/"
cp "$REPO_ROOT/persistent_data/GateMaterials.db" "$STAGE_DIR/persistent_data/"
cp -r "$REPO_ROOT/qmirt/src" "$STAGE_DIR/qmirt/"
# Editable-install and bytecode artifacts exist only on developer machines, so they
# would make the same sources hash differently depending on where you publish from.
find "$STAGE_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE_DIR" -name '*.egg-info' -type d -prune -exec rm -rf {} +
find "$STAGE_DIR" -name '*.pyc' -type f -delete

# Hash the staged file contents, not the archive: tar and gzip differ between the
# workstation and the access point, so archive bytes would give the same sources two
# different names and publish duplicates.
HASH="$(cd "$STAGE_DIR" && find . -type f -exec sha256sum {} + \
    | LC_ALL=C sort -k2 | sha256sum | cut -c1-12)"
NAME="qmirt-cardiac-payload-${HASH}.tar.gz"

TARBALL="${STAGE_DIR}.tar.gz"
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime='UTC 2020-01-01' \
    -C "$STAGE_DIR" -cf - payload persistent_data qmirt | gzip -n -9 > "$TARBALL"
SIZE_MB="$(du -m "$TARBALL" | cut -f1)"

mkdir -p "$DEST_DIR"
DEST_PATH="${DEST_DIR}/${NAME}"

if [[ -f "$DEST_PATH" && "$FORCE" -eq 0 ]]; then
    log "Already published (${SIZE_MB} MB): $DEST_PATH"
else
    # Same-name publish is atomic so a job can never fetch a half-written tarball.
    cp "$TARBALL" "${DEST_PATH}.tmp"
    mv "${DEST_PATH}.tmp" "$DEST_PATH"
    log "Published (${SIZE_MB} MB): $DEST_PATH"
fi
rm -f "$TARBALL"

log "Payload contents: payload/python, persistent_data/{cardiac_spect,GateMaterials.db}, qmirt/src"
echo "${OSDF_PREFIX}/${NAME}"
