"""Watch the brain SPECT phantom campaigns on OSPool and ERIS (harvest, check, merge).

Runs on the workstation (systemd user timer qmirt-brain-phantom-harvest, like the brain SRM
harvest). Per run:
1. OSPool: rsync the returned job tarballs of every full_* campaign
   (/ospool/ap40/data/fang.han/brain_phantom) to <local>/ospool/<campaign>/jobs/, read the
   queue (condor_q: idle / running / held, hold reasons, starts) and give held jobs a
   PeriodicRelease (up to MAX_STARTS starts).
2. ERIS: rsync every full_* campaign's slices (no ROOT files) from
   /scratch/f/fh890/brain_phantom to <local>/eris/<campaign>/, read the queue (squeue).
3. Check every slice locally: exit code, list-mode present, primaries; list failed slices
   and slices missing after their job left the queue (to resubmit).
4. Write <local>/<cluster>/<campaign>/progress_phantom.json and append a line to
   <local>/watch.log. When a campaign is complete on a cluster (every slice done with exit 0),
   merge it (merge_phantom_listmode.py) into merged/projections.npz, once.

Both clusters are reached over the SSH master connections (aliases ospool, eris); if a host
does not answer, that cluster is skipped this time and the log says so.

    python watch_brain_phantom_campaigns.py [--local-root DIR] [--no-pull] [--no-merge] [--campaign-glob 'full_*']
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "payload" / "python"))

OSPOOL_ROOT = "/ospool/ap40/data/fang.han/brain_phantom"
ERIS_ROOT = "/scratch/f/fh890/brain_phantom"
MAX_STARTS = 5
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
CONDOR_STATES = {1: "idle", 2: "running", 3: "removed", 4: "completed", 5: "held", 6: "transferring", 7: "suspended"}


def run(cmd, timeout=600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 124, ""


def remote(host, command, timeout=300) -> tuple[int, str]:
    code, out = run(SSH + [host, command], timeout)
    # ERIS prints a login banner; keep only lines that are not part of it
    lines = [l for l in out.splitlines() if not l.startswith(("|", "+")) and "Authorized access" not in l]
    return code, "\n".join(lines)


# ---------------------------------------------------------------------------------------
# OSPool
# ---------------------------------------------------------------------------------------
def ospool_campaigns(glob: str) -> dict:
    code, out = remote("ospool", f"cd {OSPOOL_ROOT} && for c in {glob}; do [ -d \"$c\" ] && echo \"$c $(cat $c/condor_submit.txt 2>/dev/null"
                                 f" | grep -o 'cluster [0-9]*' | tr -dc 0-9)\"; done")
    if code != 0:
        return {}
    return {name: int(cid) for name, cid in (l.split() for l in out.splitlines() if len(l.split()) == 2)}


def ospool_queue(clusters: list[int]) -> dict:
    if not clusters:
        return {}
    code, out = remote("ospool", "condor_q " + " ".join(map(str, clusters)) +
                       " -af ClusterId ProcId JobStatus NumJobStarts HoldReasonCode HoldReason 2>/dev/null")
    q = {}
    for line in out.splitlines():
        f = line.split(None, 5)
        if len(f) >= 4:
            q.setdefault(int(f[0]), {})[int(f[1])] = {"state": CONDOR_STATES.get(int(f[2]), f[2]), "starts": int(f[3]),
                                                      "hold_code": f[4] if len(f) > 4 else "", "hold": f[5] if len(f) > 5 else ""}
    return q


def ospool_auto_release(clusters: list[int]) -> None:
    """Held jobs come back by themselves (half an hour after holding), up to MAX_STARTS starts."""
    expr = f"(NumJobStarts < {MAX_STARTS}) && ((time() - EnteredCurrentStatus) > 1800)"
    for c in clusters:
        remote("ospool", f"condor_qedit {c} PeriodicRelease '{expr}' >/dev/null 2>&1", timeout=120)


def ospool_pull(name: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    code, _ = run(["rsync", "-a", "--ignore-existing", "-e", " ".join(SSH),
                   f"ospool:{OSPOOL_ROOT}/{name}/campaign_manifest.json", f"ospool:{OSPOOL_ROOT}/{name}/phantom_c_*.tar.gz",
                   str(dest) + "/"], timeout=3600)
    return code


def ospool_slices(dest: Path, manifest: dict) -> dict:
    """slice index -> info, from the pulled tarballs (each unpacked once, ROOT never shipped).
    The slice is slice.txt (newer jobs), else the list-mode time window, else proc + offset of
    the campaign's first cluster. A slice that succeeded in any job counts as done."""
    out = {}
    offset = int(manifest.get("slice_offset", 0))
    slice_s = float(manifest.get("slice_s", 0))
    first = int(str(manifest.get("first_cluster", "0")) or 0)
    for t in sorted(dest.glob("phantom_c_*.tar.gz")):
        cluster = int(t.name.split("_c_")[1].split("_p_")[0])
        proc = int(t.name.split("_p_")[1].split(".")[0])
        d = dest / t.name.replace(".tar.gz", "")
        if not d.exists():
            try:
                with tarfile.open(t) as tf:
                    tf.extractall(dest, filter="data")
            except (tarfile.TarError, OSError) as e:
                if cluster == first or not first:
                    out.setdefault(proc + offset, {"ok": False, "error": f"bad tarball: {e}"})
                continue
        info = slice_info(d)
        if (d / "slice.txt").exists():
            s = int((d / "slice.txt").read_text().strip())
        elif "window" in info and slice_s:
            s = int(round(info["window"][0] / slice_s))
        elif cluster == first or not first:
            s = proc + offset
        else:
            continue
        if not out.get(s, {}).get("ok"):
            out[s] = info
    return out


# ---------------------------------------------------------------------------------------
# ERIS
# ---------------------------------------------------------------------------------------
def eris_campaigns(glob: str) -> dict:
    code, out = remote("eris", f"cd {ERIS_ROOT} && for c in {glob}; do [ -d \"$c\" ] && echo \"$c $(cat $c/slurm_job_id.txt 2>/dev/null)\"; done")
    if code != 0:
        return {}
    return {l.split()[0]: int(l.split()[1]) for l in out.splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()}


def eris_queue(job_ids: list[int]) -> dict:
    if not job_ids:
        return {}
    code, out = remote("eris", "squeue -h -r -j " + ",".join(map(str, job_ids)) + " -o '%i %t %r' 2>/dev/null")
    q = {}
    for line in out.splitlines():
        f = line.split(None, 2)
        if len(f) >= 2 and "_" in f[0]:
            job, task = f[0].split("_", 1)
            if task.isdigit():
                q.setdefault(int(job), {})[int(task)] = {"state": f[1], "reason": f[2] if len(f) > 2 else ""}
    return q


def eris_pull(name: str, dest: Path) -> int:
    """ERIS has no rsync: copy the campaign's small files and every finished slice
    (exit_code.txt present) that is not here yet, as a tar stream over ssh."""
    dest.mkdir(parents=True, exist_ok=True)
    code, out = remote("eris", f"cd {ERIS_ROOT}/{name} && ls -d slices/slice_*/exit_code.txt 2>/dev/null; true")
    if code != 0:
        return code
    finished = [l.split("/")[1] for l in out.splitlines() if l.startswith("slices/slice_")]
    new = [d for d in finished if not (dest / "slices" / d / "exit_code.txt").exists()]
    items = ["campaign_manifest.json", "slurm_job_id.txt", "task.sh"] + [f"slices/{d}" for d in new]
    rc = 0
    for i in range(0, len(items), 400):  # bounded command lines
        batch = " ".join(items[i:i + 400])
        cmd = (f"cd {ERIS_ROOT}/{name} && tar -cf - --exclude='*.root' --exclude=phantom_image {batch} 2>/dev/null")
        p1 = subprocess.Popen(SSH + ["eris", cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=p1.stdout, capture_output=True, timeout=3600)
        p1.wait()
        rc = rc or p2.returncode
    return rc


def eris_slices(dest: Path) -> dict:
    out = {}
    for d in sorted((dest / "slices").glob("slice_*")):
        if (d / "exit_code.txt").exists():
            out[int(d.name.split("_")[1])] = slice_info(d)
    return out


# ---------------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------------
def slice_info(d: Path) -> dict:
    info = {"dir": str(d)}
    try:
        info["exit"] = int((d / "exit_code.txt").read_text().strip())
    except (OSError, ValueError):
        info["exit"] = None
    lm = d / "listmode.npz"
    info["ok"] = info["exit"] == 0 and lm.exists()
    if lm.exists():
        try:
            meta = json.loads(str(np.load(lm)["meta"]))
            info.update(primaries=meta["primaries"], singles=meta["singles"], window=meta["time_window_s"])
        except Exception as e:  # noqa: BLE001  (a corrupt file is a failed slice)
            info.update(ok=False, error=f"listmode unreadable: {e}")
    if (d / "job_info.json").exists():
        try:
            info["job"] = json.loads((d / "job_info.json").read_text())
        except ValueError:
            pass
    return info


def assess(name: str, cluster: str, manifest: dict, slices: dict, queue_left: int, queue_states: dict) -> dict:
    n = int(manifest.get("job_count") or manifest.get("slice_count"))
    offset = int(manifest.get("slice_offset", 0))
    expected = range(offset, offset + n)
    done = [s for s in expected if slices.get(s, {}).get("ok")]
    failed = sorted(s for s in expected if s in slices and not slices[s].get("ok"))
    missing = sorted(s for s in expected if s not in slices)
    walls = [slices[s]["job"]["wall_s"] for s in done if "job" in slices[s]]
    return {"campaign": name, "cluster": cluster, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phantom_spec": manifest.get("phantom_spec"), "activity_bq": manifest.get("activity_bq"),
            "slices_expected": n, "slices_done": len(done), "slices_failed": failed,
            "slices_missing": missing if queue_left == 0 else len(missing),
            "queue": queue_states, "queue_jobs_left": queue_left,
            "primaries_done": int(sum(slices[s].get("primaries", 0) for s in done)),
            "singles_done": int(sum(slices[s].get("singles", 0) for s in done)),
            "acquisition_s_done": round(len(done) * float(manifest.get("slice_s", 0)), 3),
            "wall_s_median": float(np.median(walls)) if walls else None,
            "complete": len(done) == n}


def merge_campaign(local: Path, slices: dict) -> str:
    out = local / "merged" / "projections.npz"
    if out.exists():
        return "merged earlier"
    files = [str(Path(s["dir"]) / "listmode.npz") for _, s in sorted(slices.items()) if s.get("ok")]
    code, _ = run([sys.executable, str(REPO / "payload" / "python" / "merge_phantom_listmode.py"), str(out), *files], timeout=6 * 3600)
    return "merged" if code == 0 and out.exists() else f"merge failed ({code})"


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--local-root", type=Path, default=Path("/data/fanghan/opengate_sim/data/brain_phantom"))
    ap.add_argument("--campaign-glob", default="full_*")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    a.local_root.mkdir(parents=True, exist_ok=True)
    log = []

    camps = ospool_campaigns(a.campaign_glob)
    if not camps:
        log.append("ospool: unreachable or no campaigns")
    else:
        queue = ospool_queue(list(camps.values()))
        ospool_auto_release([c for c in camps.values() if any(j["state"] == "held" for j in queue.get(c, {}).values())])
        for name, cid in camps.items():
            local = a.local_root / "ospool" / name
            if not a.no_pull:
                ospool_pull(name, local / "jobs")
            manifest = json.loads((local / "jobs" / "campaign_manifest.json").read_text()) | {"first_cluster": cid}
            jobs = queue.get(cid, {})
            states = {s: sum(1 for j in jobs.values() if j["state"] == s) for s in {j["state"] for j in jobs.values()}}
            holds = sorted({j["hold"][:120] for j in jobs.values() if j["state"] == "held"})
            slices = ospool_slices(local / "jobs", manifest)
            p = assess(name, "ospool", manifest, slices, len(jobs), states) | {"condor_cluster": cid, "hold_reasons": holds}
            if p["complete"] and not a.no_merge:
                p["merge"] = merge_campaign(local, slices)
            (local / "progress_phantom.json").write_text(json.dumps(p, indent=1) + "\n")
            log.append(f"ospool {name}: {p['slices_done']}/{p['slices_expected']} done, failed {len(p['slices_failed'])}, "
                       f"queue {states}{' MERGE ' + p['merge'] if 'merge' in p else ''}")

    camps = eris_campaigns(a.campaign_glob)
    if not camps:
        log.append("eris: unreachable or no campaigns")
    else:
        queue = eris_queue(list(camps.values()))
        for name, jid in camps.items():
            local = a.local_root / "eris" / name
            if not a.no_pull:
                eris_pull(name, local)
            manifest = json.loads((local / "campaign_manifest.json").read_text())
            tasks = queue.get(jid, {})
            states = {s: sum(1 for t in tasks.values() if t["state"] == s) for s in {t["state"] for t in tasks.values()}}
            # a slice of a running task has no exit code yet: count it as in the queue
            slices = eris_slices(local)
            p = assess(name, "eris", manifest, slices, len(tasks), states) | {"slurm_array": jid}
            if p["complete"] and not a.no_merge:
                p["merge"] = merge_campaign(local, slices)
            (local / "progress_phantom.json").write_text(json.dumps(p, indent=1) + "\n")
            log.append(f"eris {name}: {p['slices_done']}/{p['slices_expected']} done, failed {len(p['slices_failed'])}, "
                       f"tasks {states}{' MERGE ' + p['merge'] if 'merge' in p else ''}")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(a.local_root / "watch.log", "a") as f:
        for line in log:
            f.write(f"{stamp} {line}\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
