"""Filesystem utility helpers for qmirt_utility."""

from __future__ import annotations

from pathlib import Path


def generate_tree(dir_path: Path, prefix: str = ""):
    """Yield a visual tree structure for a directory recursively."""
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    contents = [path for path in dir_path.iterdir() if not path.name.startswith(".")]
    contents.sort(key=lambda x: (x.is_file(), x.name))

    pointers = [tee] * (len(contents) - 1) + [last] if contents else []
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():
            extension = branch if pointer == tee else space
            yield from generate_tree(path, prefix=prefix + extension)


def find_project_root(current_path: Path, marker: str = ".git") -> Path:
    """Find the nearest parent directory containing the marker path."""
    for parent_dir in [current_path] + list(current_path.parents):
        if (parent_dir / marker).exists():
            return parent_dir
    raise FileNotFoundError(f"Could not find the project root containing {marker}!")


__all__ = ["generate_tree", "find_project_root"]
