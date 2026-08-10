#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JOB_COUNT="${1:-100}"
EXECUTABLE_PATH="$SCRIPT_DIR/wrapper_cardiac_spect_sim_batch.sh"

# These paths are resolved on the submit side when condor_submit reads the file.
# The wrapper itself will run in the execute sandbox and use staged copies from $PWD.
PAYLOAD_DIR="../payload"
PERSISTENT_DATA_DIR="../persistent_data"
QMIRT_DIR="../qmirt"

if ! [[ "$JOB_COUNT" =~ ^[1-9][0-9]*$ ]]; then
	echo "Usage: $0 [job_count]"
	echo "job_count must be a positive integer"
	exit 1
fi

# 1. Generate a unique batch ID
BATCH_ID="batch_$(date +%Y%m%d_%H%M%S)"

# 2. Define the two separate directories
DATA_DIR="/ospool/ap40/data/fang.han/${BATCH_ID}"
LOG_DIR="logs/${BATCH_ID}"
SUB_FILE="${LOG_DIR}/cardiac_spect_sim_batch.sub"

# 3. Create both directories
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

cat > "$SUB_FILE" <<EOF
executable = $EXECUTABLE_PATH
arguments = \$(ClusterId) \$(ProcId)

+SingularityImage = "osdf:///ospool/ap40/data/fang.han/qmirt-gate-10-sim.sif"
+ProjectName = "MGH_Sabet"
transfer_input_files = $PAYLOAD_DIR, $PERSISTENT_DATA_DIR, $QMIRT_DIR

# Route the heavy data output to the /ospool storage directory
transfer_output_files = results_\$(ClusterId)_\$(ProcId).tar.gz
transfer_output_remaps = "results_\$(ClusterId)_\$(ProcId).tar.gz = \$(out_dir)/results_\$(ClusterId)_\$(ProcId).tar.gz"

# Route all HTCondor logs to the local /home logs directory
log = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).log
output = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).out
error = \$(log_dir)/job_\$(ClusterId)_\$(ProcId).err

request_cpus = 1
request_memory = 2GB
request_disk = 5GB

queue ${JOB_COUNT}
EOF

echo "Created data folder: $DATA_DIR"
echo "Created log folder:  $LOG_DIR"
echo "Created submit file: $SUB_FILE"

# 4. Submit the job passing both paths as variables
condor_submit "$SUB_FILE" out_dir="$DATA_DIR" log_dir="$LOG_DIR"