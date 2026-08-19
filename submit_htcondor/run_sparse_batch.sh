#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# 1. Generate a unique batch ID
BATCH_ID="sparse_batch_$(date +%Y%m%d_%H%M%S)"
FILES_PER_JOB=100
FOV_SIZE=${FOV_SIZE:-150}
INPUT_SOURCE="filenames.txt"
VOXEL_SIZES=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		-i|--input-source)
			[[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
			INPUT_SOURCE="$2"
			shift 2
			;;
		--fov-size)
			[[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
			FOV_SIZE="$2"
			shift 2
			;;
		-h|--help)
			printf 'Usage: %s [--input-source PATH] [--fov-size MM] [VOXEL_SIZE ...]\n' "$0"
			exit 0
			;;
		*)
			VOXEL_SIZES+=("$1")
			shift
			;;
	esac
done

if [[ ${#VOXEL_SIZES[@]} -eq 0 ]]; then
	VOXEL_SIZES=(1 1.5 2)
fi

# 2. Define separate data and log directories
DATA_DIR="/ospool/ap40/data/fang.han/${BATCH_ID}"
LOG_DIR="logs/${BATCH_ID}"
BATCH_DIR="sparse_input_lists/${BATCH_ID}"
INPUT_MANIFEST="${BATCH_DIR}/input_files.txt"

# 3. Create both directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$BATCH_DIR"

if [[ -d "$INPUT_SOURCE" ]]; then
	mapfile -d '' -t SOURCE_FILES < <(
		find "$INPUT_SOURCE" -maxdepth 1 -type f -name '*.tar.gz' -print0 | sort -z
	)
elif [[ -f "$INPUT_SOURCE" ]]; then
	mapfile -t SOURCE_FILES < "$INPUT_SOURCE"
else
	echo "Input source does not exist: $INPUT_SOURCE" >&2
	exit 1
fi

SOURCE_FILES=("${SOURCE_FILES[@]//[$'\r\n']/}")
if [[ ${#SOURCE_FILES[@]} -eq 0 ]]; then
	echo "No tar.gz archives found in input source: $INPUT_SOURCE" >&2
	exit 1
fi

for source_file in "${SOURCE_FILES[@]}"; do
	if [[ "$source_file" == *[,[:space:]]* ]]; then
		echo "Input paths cannot contain commas or whitespace: $source_file" >&2
		exit 1
	fi
done

JOB_COUNT=$(((${#SOURCE_FILES[@]} + FILES_PER_JOB - 1) / FILES_PER_JOB))
: > "$INPUT_MANIFEST"

for ((job_index = 0; job_index < JOB_COUNT; job_index++)); do
	start_index=$((job_index * FILES_PER_JOB))
	slice=("${SOURCE_FILES[@]:start_index:FILES_PER_JOB}")
	[[ ${#slice[@]} -gt 0 ]] || break
	(IFS=,; printf '%s\n' "${slice[*]}") >> "$INPUT_MANIFEST"
done

echo "Created data folder: $DATA_DIR"
echo "Created log folder:  $LOG_DIR"
echo "Created input manifest: $INPUT_MANIFEST"
echo "Discovered ${#SOURCE_FILES[@]} archives in $JOB_COUNT jobs from $INPUT_SOURCE"

# 4. Submit one worker per archive slice. Each worker writes all resolutions.
echo "Submitting ${#VOXEL_SIZES[@]} SRM resolutions in $JOB_COUNT jobs to $DATA_DIR"
condor_submit batch_sparse.sub \
	out_dir="$DATA_DIR" \
	log_dir="$LOG_DIR" \
	input_manifest="$INPUT_MANIFEST" \
	voxel_sizes="${VOXEL_SIZES[*]}" \
	fov_size="$FOV_SIZE"
