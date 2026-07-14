"""Filesystem utility helpers for qmirt_utility."""

from __future__ import annotations

import inspect
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

def search_dir_up(dir_name: str, start_file: str | Path) -> Path:
    """
    Search for a directory name by traversing up the directory tree.
    
    Args:
        dir_name: The name of the target directory to find.
        start_file: The path of the file to start searching from (typically __file__).
        
    Returns:
        Path: The absolute path of the found directory.
        
    Raises:
        FileNotFoundError: If the directory is not found before reaching the root.
    """
    # Extract the parent directory from the provided file path
    current_dir = Path(start_file).resolve().parent
    
    while True:
        target = current_dir / dir_name
        
        # If the target exists and is a directory, return its path
        if target.is_dir():
            return target
            
        parent_dir = current_dir.parent
        
        # Stop if we have reached the root of the file system
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Directory '{dir_name}' not found in any parent directories of '{start_file}'.")
            
        # Move up to the next parent directory
        current_dir = parent_dir


def find_project_root(start_path: str | Path = None, marker: str = ".git") -> Path:
    """
    Find the nearest parent directory containing the marker path.
    Automatically detects starting path for scripts and Jupyter notebooks if not provided.
    """
    # --- Step 1: Determine the starting path ---
    if start_path is not None:
        # Requirement 3: User input is optional. Use it if provided.
        current_path = Path(start_path).resolve()
    else:
        # Inspect the caller's environment
        caller_globals = inspect.stack()[1][0].f_globals

        # Requirement 1: Command line interface python script call
        if "__file__" in caller_globals:
            current_path = Path(caller_globals["__file__"]).resolve().parent

        # Requirement 2: Jupyter notebooks (VS Code specific environment)
        elif "__vsc_ipynb_file__" in caller_globals:
            current_path = Path(caller_globals["__vsc_ipynb_file__"]).resolve().parent

        # Requirement 2: Standard Jupyter/JupyterLab fallback
        else:
            try:
                # Requires `pip install ipynbname` to strictly avoid the running env (cwd)
                import ipynbname

                current_path = ipynbname.path().parent
            except ImportError:
                print(
                    "Warning: 'ipynbname' not installed. Could not strictly determine notebook path. Falling back to CWD."
                )
                current_path = Path.cwd()

    # --- Step 2: Find the project root ---
    for parent_dir in [current_path] + list(current_path.parents):
        if (parent_dir / marker).exists():
            return parent_dir

    raise FileNotFoundError(
        f"Could not find the project root containing '{marker}' starting from {current_path}!"
    )


__all__ = ["generate_tree", "find_project_root"]
