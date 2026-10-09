"""Reduce one brain SPECT phantom job (gate_sim_brain_spect_boolean.py --mode phantom) to
compact list-mode.

Input: the job's output directory (pixel_singles_*.root with the merged "Singles" tree,
written without energy blur, *_sim_stats.txt, *_run_manifest.json).
Output: listmode.npz with one row per single
    head              uint8    0-based head (same index as the SRM's "crystal")
    pixel             uint16   0..624 pixel of the head (same as the SRM's "pixel")
    energy_kev        float32  true deposited energy (blur later: merge_phantom_listmode.py)
    time_s            float32  GlobalTime (s from the acquisition start)
    phantom_compton   uint8    Compton steps in the phantom along the photon's history
    phantom_rayleigh  uint8    Rayleigh steps in the phantom
    phantom_scatter   uint8    1 if the photon scattered in the phantom: Compton or Rayleigh
                               steps there and, when the hardware was in a parallel world
                               (image phantoms), the last interaction in the image box in a
                               tissue voxel (else the steps were in hardware inside the box)
    event_mm          float32 (N, 3)  decay position (scanner frame)
and meta (JSON): primaries, time window, seed, manifest. Singles below --min-kev are dropped.

    python reduce_phantom_singles.py JOB_DIR [--out JOB_DIR/listmode.npz] [--min-kev 10]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

HEAD_PIXEL = re.compile(r"^pixel_(\d+)_param.*_(\d+)$")
BRANCHES = ["PreStepUniqueVolumeID", "TotalEnergyDeposit", "GlobalTime",
            "EventPosition_X", "EventPosition_Y", "EventPosition_Z"]


def scatter_branches(keys) -> tuple[str, str]:
    """Phantom Compton / Rayleigh count columns: PhantomCompton/PhantomRayleigh (opengate
    >= 10.1.1 auxiliary attributes) or ProcessDefinedStep__compt__<volume> /
    ProcessDefinedStep__Rayl__<volume> (10.1.0 actor-based attributes)."""
    if "PhantomCompton" in keys:
        return "PhantomCompton", "PhantomRayleigh"
    compt = [k for k in keys if k.startswith("ProcessDefinedStep__compt__")]
    rayl = [k for k in keys if k.startswith("ProcessDefinedStep__Rayl__")]
    if len(compt) != 1 or len(rayl) != 1:
        raise SystemExit(f"phantom scatter columns not found in {sorted(keys)}")
    return compt[0], rayl[0]
LAST = ["PhantomLastInteraction_X", "PhantomLastInteraction_Y", "PhantomLastInteraction_Z"]


def _tissue_at(spec_file: str, world_mm: np.ndarray) -> np.ndarray:
    """True where a world point is in a tissue voxel (label > 0) of the image phantom."""
    import phantom_models as pm

    spec = pm.Spec.from_dict(json.loads(Path(spec_file).read_text()))
    labels, _ = pm.sample_image(spec, world_mm)
    return np.asarray(labels) > 0


def _head_pixel(volume_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = [v.decode() if isinstance(v, bytes) else str(v) for v in volume_ids]
    uniq, inv = np.unique(ids, return_inverse=True)
    hp = np.array([[int(g) for g in HEAD_PIXEL.match(u).groups()] for u in uniq], dtype=np.int64).reshape(-1, 2)
    return (hp[inv, 0] - 1).astype(np.uint8), hp[inv, 1].astype(np.uint16)


def reduce_job(job_dir: Path, out: Path, min_kev: float = 10.0) -> dict:
    import uproot

    roots = sorted(job_dir.glob("pixel_singles_*.root"))
    if len(roots) != 1:
        raise SystemExit(f"{job_dir}: expected one pixel_singles_*.root, found {len(roots)}")
    manifest = json.loads(next(job_dir.glob("*_run_manifest.json")).read_text())
    cols = {k: [] for k in ("head", "pixel", "energy_kev", "time_s", "phantom_compton", "phantom_rayleigh",
                            "phantom_scatter", "event_mm")}
    with uproot.open(roots[0]) as f:
        tree = f["Singles"]
        has_last = LAST[0] in tree.keys()
        c_key, r_key = scatter_branches(tree.keys())
        for d in tree.iterate(BRANCHES + [c_key, r_key] + (LAST if has_last else []), library="np", step_size="200 MB"):
            d["PhantomCompton"], d["PhantomRayleigh"] = d.pop(c_key), d.pop(r_key)
            e = d["TotalEnergyDeposit"] * 1000.0
            keep = e >= min_kev
            head, pixel = _head_pixel(d["PreStepUniqueVolumeID"][keep])
            cols["head"].append(head)
            cols["pixel"].append(pixel)
            cols["energy_kev"].append(e[keep].astype(np.float32))
            cols["time_s"].append((d["GlobalTime"][keep] * 1e-9).astype(np.float32))
            cols["phantom_compton"].append(np.minimum(d["PhantomCompton"][keep], 255).astype(np.uint8))
            cols["phantom_rayleigh"].append(np.minimum(d["PhantomRayleigh"][keep], 255).astype(np.uint8))
            cols["event_mm"].append(np.stack([d[f"EventPosition_{a}"][keep] for a in "XYZ"], 1).astype(np.float32))
            scattered = (d["PhantomCompton"][keep] + d["PhantomRayleigh"][keep]) > 0
            if has_last and scattered.any():
                last = np.stack([d[k][keep] for k in LAST], 1)
                idx = np.flatnonzero(scattered)
                ok = np.isfinite(last[idx]).all(1)
                in_tissue = np.zeros(len(idx), bool)
                in_tissue[ok] = _tissue_at(manifest["parameters"]["phantom_spec"], last[idx][ok])
                scattered[idx] = in_tissue
            cols["phantom_scatter"].append(scattered.astype(np.uint8))
    arrays = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    stats = json.loads(next(job_dir.glob("*_sim_stats.txt")).read_text())
    p = manifest["parameters"]
    meta = {"schema": "qmirt.phantom_listmode/1", "job_dir": str(job_dir), "singles": int(len(arrays["head"])),
            "min_kev": min_kev, "primaries": int(stats["events"]["value"] if isinstance(stats["events"], dict) else stats["events"]),
            "time_window_s": [p["time_start_s"], p["time_start_s"] + p["chunk_duration_s"] * p["num_chunks"]],
            "random_seed": manifest["random_seed"], "manifest": manifest}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta, default=str)), **arrays)
    return meta


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-kev", type=float, default=10.0)
    a = ap.parse_args()
    m = reduce_job(a.job_dir, a.out or a.job_dir / "listmode.npz", a.min_kev)
    print(f"{m['singles']:,} singles from {m['primaries']:,} primaries, window {m['time_window_s']} s")


if __name__ == "__main__":
    main()
