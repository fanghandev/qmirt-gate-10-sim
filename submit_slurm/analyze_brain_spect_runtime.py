#!/usr/bin/env python3

import json
import re
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("/home/fanghan/PHShome/scratch/brain_spect_sim")
OUT_DIR = Path("/home/fanghan/Work/RPIL/QMIRT/qmirt-gate-10-sim/results/brain_spect")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_stats_rows():
    rows = []
    for stats_path in sorted(BASE_DIR.rglob("a_*_sim_stats.txt")):
        try:
            with stats_path.open() as fh:
                data = json.load(fh)
            events = float(data["events"]["value"])
            duration = float(data["duration"]["value"])
            pps = float(data["pps"]["value"])
            rows.append(
                {
                    "path": stats_path,
                    "events": events,
                    "duration": duration,
                    "pps": pps,
                }
            )
        except Exception as exc:
            print(f"Skipping {stats_path}: {exc}")
    return rows


def load_wall_times():
    wall_by_task = {}
    for wall_path in sorted(BASE_DIR.rglob("task_*_wall_time.txt")):
        m = re.search(r"task_(\d+)_wall_time\.txt$", str(wall_path))
        if not m:
            continue
        task_id = int(m.group(1))
        try:
            text = wall_path.read_text()
            m2 = re.search(r"wall_time_seconds:\s*(\d+)", text)
            if m2:
                wall_by_task[task_id] = int(m2.group(1))
        except Exception as exc:
            print(f"Skipping {wall_path}: {exc}")
    return wall_by_task


def build_summary():
    stats_rows = load_stats_rows()
    wall_times = load_wall_times()
    rows = []
    for row in stats_rows:
        # infer task id from filename like a_3850338_j_0_sim_stats.txt
        m = re.search(r"_j_(\d+)_sim_stats\.txt$", row["path"].name)
        task_id = int(m.group(1)) if m else None
        wall = wall_times.get(task_id)
        rows.append(
            {
                **row,
                "wall_time": wall,
                "overhead": (wall - row["duration"]) if wall is not None else None,
            }
        )

    rows = [r for r in rows if r["wall_time"] is not None]
    if not rows:
        raise RuntimeError(
            "No valid task rows with both sim stats and wall time found."
        )

    summary = {
        "n_tasks": len(rows),
        "events_mean": mean(r["events"] for r in rows),
        "events_min": min(r["events"] for r in rows),
        "events_max": max(r["events"] for r in rows),
        "duration_mean": mean(r["duration"] for r in rows),
        "wall_mean": mean(r["wall_time"] for r in rows),
        "overhead_mean": mean(r["overhead"] for r in rows if r["overhead"] is not None),
        "pps_mean": mean(r["pps"] for r in rows),
        "total_events": sum(r["events"] for r in rows),
    }
    return rows, summary


def make_plot(rows, x_key, y_key, title, ylabel, out_name):
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.array([r[y_key] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=32, alpha=0.8, color="#2F6FED")
    ax.set_title(title)
    ax.set_xlabel("Number of simulated events")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / out_name, dpi=200)
    plt.close(fig)


def main():
    rows, summary = build_summary()
    csv_path = OUT_DIR / "brain_spect_runtime_summary.csv"
    with csv_path.open("w") as fh:
        fh.write("events,duration,wall_time,overhead,pps\n")
        for r in rows:
            fh.write(
                f"{r['events']},{r['duration']},{r['wall_time']},{r['overhead']},{r['pps']}\n"
            )

    make_plot(
        rows,
        x_key="events",
        y_key="wall_time",
        title="Task wall time vs number of simulated events",
        ylabel="Wall time (s)",
        out_name="task_time_vs_events.png",
    )
    make_plot(
        rows,
        x_key="events",
        y_key="duration",
        title="Pure simulation time vs number of simulated events",
        ylabel="Simulation duration (s)",
        out_name="pure_simulation_time_vs_events.png",
    )

    print("Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"CSV written to: {csv_path}")
    print(f"Plots written to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
