# qmirt-utility

This project provides an installable package `qmirt_utility` with utilities to parse and plot WRL geometry used in the repository.

Install locally (editable):

python -m pip install -e dev/python/qmirt-utility

Usage:

from qmirt_utility.plotting import plot_wrl_file

fig = plot_wrl_file("path/to/sim_geometry.wrl")

Track plotting helpers live in `qmirt_utility.plotting.tracks`.

To override or ignore stylesheet `exclude_patterns`, pass `exclude_mode`:

fig = plot_wrl_file(
"path/to/sim_geometry.wrl",
style_sheet="wrl_stylesheet.example.json",
exclude_patterns=["world:*"],
exclude_mode="replace",
)
