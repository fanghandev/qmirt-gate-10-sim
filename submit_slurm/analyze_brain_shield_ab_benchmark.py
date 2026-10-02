#!/usr/bin/env python3
"""Cost per primary and production loop sizing from the Expanse A/B benchmark.

Reads the campaign directories written by launch_brain_expanse_shield_ab_benchmark.sh
(one task, one loop each). For every configuration (actor layout, shield model) the
two loop sizes give

  task wall time = fixed + primaries_per_thread * cost_per_primary_per_thread

(the fixed part covers Gate start-up, geometry build and the ROOT to SRM conversion
of the loop, which scales weakly with size). From that it recommends NUM_CHUNKS,
NUM_LOOPS and SU_PER_LOOP for the production launchers.

Usage: python3 analyze_brain_shield_ab_benchmark.py <campaign dir> [<campaign dir> ...]
       [--loop-minutes 45] [--time-limit-hours 24] [--fill 0.75]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

EVENTS_PER_CHUNK_PER_THREAD = 6.25e6  # production activity x 1 s chunks


def _duration_s(value):
    scale = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}
    return float(value["value"]) * scale.get(value.get("unit", "s"), 1.0)


def load_campaign(path):  # plain hints: Expanse has Python 3.6
    manifest_path = os.path.join(path, "campaign_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"skip {path}: no campaign_manifest.json")
        return None
    manifest = json.load(open(manifest_path))
    threads = int(manifest["cpus_per_task"])
    # per-task means (the benchmark uses one task; production campaigns have many)
    primaries, sim_s, n_tasks, wall_s = 0.0, 0.0, 0, []
    # progress.json on Expanse; cluster_progress.json in copies pulled by Globus
    progress = next((os.path.join(path, n) for n in ("progress.json", "cluster_progress.json")
                     if os.path.exists(os.path.join(path, n))), "")
    if progress:
        tasks = [t for t in json.load(open(progress)).get("per_task", []) if t.get("primaries")]
        n_tasks = len(tasks)
        primaries = sum(float(t["primaries"]) for t in tasks)
        sim_s = sum(float(t.get("simulation_seconds") or 0) for t in tasks)
    if primaries == 0:  # fall back to the per-loop Gate statistics files
        stats_files = glob.glob(os.path.join(path, "**", "*_sim_stats_loop_*.txt"), recursive=True)
        n_tasks = len({os.path.dirname(f) for f in stats_files})
        for stats in stats_files:
            d = json.load(open(stats))
            primaries += float(d["events"]["value"])
            sim_s += _duration_s(d["duration"])
    for wall in glob.glob(os.path.join(path, "**", "task_*_wall_time.txt"), recursive=True):
        m = re.search(r"wall_time_seconds:\s*([0-9.]+)", open(wall).read())
        if m:
            wall_s.append(float(m.group(1)))
    if primaries == 0:
        print(f"skip {path}: no finished loop found")
        return None
    n_tasks = max(n_tasks, 1)
    return {
        "path": path,
        "actor_layout": manifest.get("actor_layout", "per-head"),
        "shield_model": manifest.get("shield_model", "stl"),
        "num_chunks": int(manifest["num_chunks"]),
        "threads": threads,
        "primaries": primaries / n_tasks,
        "simulation_s": sim_s / n_tasks,
        "wall_s": sum(wall_s) / len(wall_s) if wall_s else sim_s / n_tasks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign_dirs", nargs="+")
    ap.add_argument("--loop-minutes", type=float, default=45.0,
                    help="target wall time of one production loop")
    ap.add_argument("--time-limit-hours", type=float, default=24.0)
    ap.add_argument("--fill", type=float, default=0.75,
                    help="fraction of the time limit to plan for (margin for slow nodes)")
    args = ap.parse_args()

    groups = defaultdict(list)
    for path in args.campaign_dirs:
        c = load_campaign(path)
        if c:
            groups[(c["actor_layout"], c["shield_model"])].append(c)

    print(f"{'actors':9s} {'shield':7s} {'us/primary/thread':>18s} {'fixed per loop (s)':>19s}"
          f" {'Gate-only us/primary/thread':>28s}")
    results = {}
    for key, runs in sorted(groups.items()):
        runs.sort(key=lambda c: c["primaries"])
        if len({c["num_chunks"] for c in runs}) < 2:
            print(f"{key[0]:9s} {key[1]:7s} need two different NUM_CHUNKS ({len(runs)} run(s) found)")
            continue
        a, b = runs[0], runs[-1]
        per_thread = lambda c: c["primaries"] / c["threads"]  # noqa: E731
        cost = (b["wall_s"] - a["wall_s"]) / (per_thread(b) - per_thread(a))
        fixed = a["wall_s"] - per_thread(a) * cost
        gate_cost = (b["simulation_s"] - a["simulation_s"]) / (per_thread(b) - per_thread(a))
        results[key] = (cost, fixed, b["threads"])
        print(f"{key[0]:9s} {key[1]:7s} {cost * 1e6:18.2f} {fixed:19.0f} {gate_cost * 1e6:28.2f}")

    if ("merged", "csg") in results:
        cost, fixed, threads = results[("merged", "csg")]
        chunk_s = EVENTS_PER_CHUNK_PER_THREAD * cost
        chunks = max(1, round((args.loop_minutes * 60 - fixed) / chunk_s))
        loop_s = fixed + chunks * chunk_s
        loops = max(1, int(args.time_limit_hours * 3600 * args.fill // loop_s))
        su_per_loop = threads * loop_s / 3600
        primaries_per_loop = chunks * EVENTS_PER_CHUNK_PER_THREAD * threads
        print(f"\nProduction recommendation (merged actors + CSG shield, {threads} threads):")
        print(f"  NUM_CHUNKS={chunks}  (loop ~{loop_s / 60:.0f} min, {primaries_per_loop:.3e} primaries)")
        print(f"  NUM_LOOPS={loops}    (task ~{loops * loop_s / 3600:.1f} h of a {args.time_limit_hours:.0f} h limit)")
        print(f"  SU_PER_LOOP={su_per_loop:.0f}")
        target = 1.016e14
        print(f"  full 288 mm target {target:.3e} primaries: ~{target / primaries_per_loop * su_per_loop:,.0f} SU")
        if ("per-head", "stl") in results:
            print(f"  speed-up vs the legacy setup: {results[('per-head', 'stl')][0] / cost:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
