#!/usr/bin/env python3
"""Split this job's CPUs into NUMA-local groups, one Gate process per group.

A single multithreaded Gate process spread over all 8 NUMA domains of an Expanse
node reads Geant4's shared geometry and physics tables from one domain's memory;
one process per domain, bound to its cores and memory, ran 3.3x faster per node
(dev/python/brain_shield_csg.md). This prints the groups for the wrapper:

    <memory nodes or -> <cpu list> <cpu count>

one line per group. "-" means no memory binding. The wrapper runs it inside the
Singularity image, which sees the host's /sys topology and the job's CPU affinity.

Usage: numa_layout.py [--split auto|off|N] [--min-cpus 4] [--cpus 0-126]
                      [--sys-root /sys]
  auto  one group per NUMA node holding the job's CPUs (one group = no split)
  off   one group with all CPUs
  N     N near-equal groups of consecutive CPUs (for testing on one-node machines)
"""

import argparse
import glob
import os
import re


def parse_cpu_list(text):
    cpus = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_cpu_list(cpus):
    cpus = sorted(cpus)
    ranges, start, prev = [], None, None
    for c in cpus:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            ranges.append((start, prev))
            start = prev = c
    if start is not None:
        ranges.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in ranges)


def numa_nodes(sys_root):
    nodes = {}
    for path in glob.glob(os.path.join(sys_root, "devices/system/node/node*/cpulist")):
        m = re.search(r"node(\d+)/cpulist$", path)
        if m:
            with open(path) as handle:
                nodes[int(m.group(1))] = parse_cpu_list(handle.read())
    return nodes


def node_of(cpu, nodes):
    for node, members in nodes.items():
        if cpu in members:
            return node
    return None


def layout(allowed, nodes, split, min_cpus):
    """List of (memory node set or None, cpu set)."""
    if split == "off" or not allowed:
        return [(None, set(allowed))]
    if split == "auto":
        groups = []
        for node in sorted(nodes):
            cpus = allowed & nodes[node]
            if cpus:
                groups.append(({node}, cpus))
        stray = allowed - set().union(*(c for _, c in groups)) if groups else set(allowed)
        if stray:  # CPUs not listed under any node (unusual): keep them, unbound
            groups.append((None, stray))
        # fold groups too small to be worth a process into a neighbour
        merged = []
        for mem, cpus in groups:
            if merged and len(cpus) < min_cpus:
                prev_mem, prev_cpus = merged[-1]
                merged[-1] = (None if prev_mem is None or mem is None else prev_mem | mem,
                              prev_cpus | cpus)
            else:
                merged.append((mem, cpus))
        if len(merged) > 1 and len(merged[0][1]) < min_cpus:
            mem0, cpus0 = merged.pop(0)
            mem1, cpus1 = merged[0]
            merged[0] = (None if mem0 is None or mem1 is None else mem0 | mem1, cpus0 | cpus1)
        if len(merged) == 1:
            return [(None, merged[0][1])]
        return merged
    n = int(split)
    if n < 1:
        raise SystemExit("--split N must be >= 1")
    ordered = sorted(allowed)
    n = min(n, len(ordered))
    groups = []
    for i in range(n):
        cpus = set(ordered[len(ordered) * i // n: len(ordered) * (i + 1) // n])
        owners = {node_of(c, nodes) for c in cpus}
        mem = owners if len(owners) == 1 and None not in owners else None
        groups.append((mem, cpus))
    if n == 1:
        return [(None, groups[0][1])]
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="auto", help="auto, off or a number of groups")
    ap.add_argument("--min-cpus", type=int, default=4)
    ap.add_argument("--cpus", default=None, help="override the CPU affinity (testing)")
    ap.add_argument("--sys-root", default="/sys", help="sysfs root (testing)")
    args = ap.parse_args()
    if args.split not in ("auto", "off") and not args.split.isdigit():
        ap.error("--split must be auto, off or a positive integer")
    allowed = parse_cpu_list(args.cpus) if args.cpus else set(os.sched_getaffinity(0))
    for mem, cpus in layout(allowed, numa_nodes(args.sys_root), args.split, args.min_cpus):
        mem_text = "-" if not mem else ",".join(str(m) for m in sorted(mem))
        print("%s %s %d" % (mem_text, format_cpu_list(cpus), len(cpus)))


if __name__ == "__main__":
    main()
