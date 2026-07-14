import argparse
import glob
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, vstack, save_npz


def merge_head_srms(input_dir: Path, output_path: Path):
    """Merge all per-head SRM files in *input_dir* into a single CSR matrix.

    Expected input files match the pattern ``srm_*_h_*.npz`` and each file
    contains a scipy.sparse CSR matrix saved with ``save_npz``.
    The resulting matrix is a vertical stack of the individual head matrices
    (i.e., shape ``(80*625, num_voxels)`` for the standard 80‑head setup).
    """
    pattern = str(input_dir / "srm_*_h_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No SRM files found in {input_dir} matching pattern {pattern}")

    matrices = []
    for f in files:
        mat = load_npz(f)
        matrices.append(mat)
        print(f"Loaded {Path(f).name} with shape {mat.shape}")

    combined = vstack(matrices, format="csr")
    combined.eliminate_zeros()
    save_npz(output_path, combined)
    print(f"Merged {len(matrices)} SRM files into {output_path.name} (shape {combined.shape})")


def main():
    parser = argparse.ArgumentParser(description="Merge per‑head sparse SRM .npz files into a single matrix")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Directory containing per‑head SRM files")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path for the merged SRM .npz file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    merge_head_srms(input_dir, output_path)

if __name__ == "__main__":
    main()
