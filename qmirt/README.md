# qmirt

This project provides an installable package `qmirt` with utilities to parse and plot WRL geometry used in the repository.

Install locally (editable):

```{shell}
python -m pip install -e dev/python/qmirt
```

Usage:

```{python}
from qmirt.plotting import plot_wrl_file

fig = plot_wrl_file("path/to/sim_geometry.wrl")
```

Track plotting helpers live in `qmirt.plotting.tracks`.

To override or ignore stylesheet `exclude_patterns`, pass `exclude_mode`:

```{python}
fig = plot_wrl_file(
"path/to/sim_geometry.wrl",
style_sheet="wrl_stylesheet.example.json",
exclude_patterns=["world:*"],
exclude_mode="replace",
)
```
