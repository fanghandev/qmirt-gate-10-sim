#!/usr/bin/env python3
"""
test_job_task_identifiers.py
=============================
Payload script that tests and prints SLURM job and task identifiers.

Usage:
  ./test_job_task_identifiers.py

Or via sbatch-wrapper:
  sbatch-wrapper.sh -N "slurm-id-test" -p normal -a "0-2" -o ./logs -- ./test_job_task_identifiers.py

Output will be saved to $OUTPUT_DIR (set by sbatch-wrapper)
"""

import os
import sys
from pathlib import Path


def get_slurm_var(var_name: str) -> str:
    """Get SLURM environment variable or return 'unset'."""
    return os.environ.get(var_name, "unset")


def get_effective_job_id() -> str:
    """Get effective job ID (array job ID or regular job ID)."""
    return os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "nojob"


def get_effective_task_id() -> str:
    """Get effective task ID (array task ID or SLURM_PROCID or 0)."""
    return os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"


def get_effective_array_job_id() -> str:
    """Get effective array job ID."""
    return os.environ.get("SLURM_ARRAY_JOB_ID", "unset")


def print_identifier_block(output_file, label: str) -> None:
    """Print a block of SLURM identifiers."""
    block = f"""[{label}]
SLURM_JOB_ID={get_slurm_var('SLURM_JOB_ID')}
SLURM_JOB_NAME={get_slurm_var('SLURM_JOB_NAME')}
SLURM_JOB_NODELIST={get_slurm_var('SLURM_JOB_NODELIST')}
SLURM_NTASKS={get_slurm_var('SLURM_NTASKS')}
SLURM_PROCID={get_slurm_var('SLURM_PROCID')}
SLURM_LOCALID={get_slurm_var('SLURM_LOCALID')}
SLURM_ARRAY_JOB_ID={get_slurm_var('SLURM_ARRAY_JOB_ID')}
SLURM_ARRAY_TASK_ID={get_slurm_var('SLURM_ARRAY_TASK_ID')}
SLURM_ARRAY_TASK_COUNT={get_slurm_var('SLURM_ARRAY_TASK_COUNT')}
SLURM_ARRAY_TASK_MIN={get_slurm_var('SLURM_ARRAY_TASK_MIN')}
SLURM_ARRAY_TASK_MAX={get_slurm_var('SLURM_ARRAY_TASK_MAX')}
SLURM_STEP_ID={get_slurm_var('SLURM_STEP_ID')}
SLURM_STEP_NUM_TASKS={get_slurm_var('SLURM_STEP_NUM_TASKS')}
SLURM_STEP_TASKS={get_slurm_var('SLURM_STEP_TASKS')}
effective_job_id={get_effective_job_id()}
effective_task_id={get_effective_task_id()}
effective_array_job_id={get_effective_array_job_id()}
"""
    output_file.write(block)
    print(block, end="")


def main():
    """Main entry point."""
    # Set output directory
    output_dir = os.environ.get("OUTPUT_DIR", "./output")
    
    # Determine report file path
    report_job_id = get_effective_job_id()
    report_task_id = os.environ.get("SLURM_PROCID", "0")
    report_file = os.path.join(output_dir, f"slurm_identifier_report_job_{report_job_id}_task_{report_task_id}.txt")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open report file for appending
    try:
        with open(report_file, "a") as output_file:
            # Write and print payload execution started
            msg = "=== Payload execution started ===\n\n"
            output_file.write(msg)
            print(msg, end="")
            
            # Write and print report file and output directory info
            info = f"report_file={report_file}\noutput_dir={output_dir}\n\n"
            output_file.write(info)
            print(info, end="")
            
            # Write and print SLURM identifier test
            msg = "=== Slurm identifier test ===\n"
            output_file.write(msg)
            print(msg, end="")
            
            print_identifier_block(output_file, "batch-shell")
            
            # Check if multi-task job
            slurm_ntasks = os.environ.get("SLURM_NTASKS", "1")
            try:
                ntasks_int = int(slurm_ntasks)
                is_multi_task = ntasks_int > 1
            except ValueError:
                is_multi_task = False
            
            if is_multi_task:
                msg = """=== Per-task step view from wrapper-launched srun ===
This payload does not launch srun itself; the wrapper provides the task context.
SLURM_PROCID={procid}
SLURM_LOCALID={localid}
SLURM_STEP_ID={step_id}
SLURM_STEP_NUM_TASKS={step_num_tasks}
effective_job_id={eff_job_id}
effective_task_id={eff_task_id}

""".format(
                    procid=get_slurm_var("SLURM_PROCID"),
                    localid=get_slurm_var("SLURM_LOCALID"),
                    step_id=get_slurm_var("SLURM_STEP_ID"),
                    step_num_tasks=get_slurm_var("SLURM_STEP_NUM_TASKS"),
                    eff_job_id=get_effective_job_id(),
                    eff_task_id=get_effective_task_id(),
                )
            else:
                msg = """=== Single-task job ===
Wrapper-launched srun still provides the single task context.

"""
            
            output_file.write(msg)
            print(msg, end="")
            
            # Write and print scenario hints
            hints = """=== Scenario hints ===
non-array job: SLURM_JOB_ID is set, SLURM_ARRAY_JOB_ID/SLURM_ARRAY_TASK_ID are unset
array element: SLURM_JOB_ID is set, SLURM_ARRAY_JOB_ID equals the array job id, SLURM_ARRAY_TASK_ID is the element index
multi-task step: SLURM_PROCID is set inside srun-launched tasks
"""
            output_file.write(hints)
            print(hints, end="")
            
    except IOError as e:
        print(f"Error writing to report file {report_file}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
