# Wrapper follow-up todo

## 2) Generic custom-wrapper metadata handling

- [ ] When a custom wrapper path is passed, derive a sensible `SIM_LABEL`, `SIM_TYPE`, `OUTPUT_SUBDIR`, and `SIM_PYTHON_SCRIPT` from the wrapper name or a provided override instead of relying on the Brain defaults.
- [ ] Ensure the generated batch metadata stays consistent even when the user runs an arbitrary wrapper script outside the repo’s built-in Brain/Cardiac cases.

## 3) Forward extra wrapper arguments

- [ ] Add a way to pass through custom runtime arguments from the launcher into the generated sbatch job so a generic wrapper can accept additional flags without hard-coding task-specific assumptions.
- [ ] Decide on the argument convention for wrapper-specific options, e.g. either a direct passthrough list or an explicit `--wrapper-arg` syntax.

## Status

- [x] Cardiac wrapper updated to the Apptainer repo-bind flow.
