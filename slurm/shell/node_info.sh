#!/bin/bash

# Print Header
printf "%-20s %-12s %-12s %-12s %-10s %-15s %-10s\n" \
"NODE" "PARTITION" "STATE" "CPUS(A/T)" "RAM(T)" "RAM(FREE)" "GRES"
echo "-------------------------------------------------------------------------------------------------------"

# Get all nodes and their partitions
# We use sinfo to get the base list
sinfo -N -h -o "%N %P %T" | while read -r NODE PART STATE; do
    
    # Use scontrol to get the nitty-gritty details for each node
    # We extract: Total CPUs, Allocated CPUs, RealMemory, FreeMem, and GRES
    NODE_DETAILS=$(scontrol show node "$NODE")
    
    CPUTOT=$(echo "$NODE_DETAILS" | grep -oP 'CPUTot=\K\d+')
    CPUALL=$(echo "$NODE_DETAILS" | grep -oP 'CPUAlloc=\K\d+')
    
    # Memory is in MB, converting to GB for readability
    MEMTOT_MB=$(echo "$NODE_DETAILS" | grep -oP 'RealMemory=\K\d+')
    MEMTOT_GB=$(( MEMTOT_MB / 1024 ))
    
    MEMFREE_MB=$(echo "$NODE_DETAILS" | grep -oP 'FreeMem=\K\d+')
    # Handle cases where FreeMem might be null or missing
    if [[ -z "$MEMFREE_MB" ]]; then
        MEMFREE_GB="N/A"
    else
        MEMFREE_GB=$(( MEMFREE_MB / 1024 ))
    fi
    
    GRES=$(echo "$NODE_DETAILS" | grep -oP 'Gres=\K\S+')
    if [[ "$GRES" == "(null)" ]]; then GRES="None"; fi

    # Print the formatted row
    printf "%-20s %-12s %-12s %-12s %-10s %-15s %-10s\n" \
    "$NODE" "$PART" "$STATE" "$CPUALL/$CPUTOT" "${MEMTOT_GB}G" "${MEMFREE_GB}G" "$GRES"

done
