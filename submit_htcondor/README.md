# How to Use

1. Make sure `filenames.txt` in `submit_htcondor/` points at the worker `results_*.tar.gz` archives on the access point.
2. Run `run_sparse_batch.sh` to stage those tar.gz inputs into 100 worker jobs and submit the batch.
3. After the workers finish, run `run_sparse_combine.sh` with the worker output directory and your final output directory. It will stage the worker tarballs into the combine job before combining them.
4. Plot the final per-crystal files with `plot_per_crystal_srm_xy.py`.
