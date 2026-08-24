# SRM Monitor

The dashboard is served by `serve_monitor.py` and displays the detector pixel mapping used by the selected-pixel SRM views.

## Default Pixel Mapping

The displayed pixel ID mapping is accurate with all three transforms enabled by default:

- **Swap row/column**: enabled
- **Flip row**: enabled
- **Flip column**: enabled

For a detector with a `25 x 25` pixel face, pixel ID is decoded as:

```text
row = floor(pixel_id / 25)
column = pixel_id % 25
```

The dashboard then applies the enabled swap and row/column flips when calculating the pixel center from the detector geometry. These defaults match the validated physical pixel orientation for the current cardiac SPECT geometry.

## Module and Geometry IDs

SRM module IDs are one-based (`1, 2, 3, ...`). Geometry records use zero-based `crystal_id` values (`0, 1, 2, ...`) while retaining one-based Gate `solid_id` values. The dashboard therefore looks up the geometry record for SRM module `m` using:

```text
geometry.crystal_id = m - 1
```

This keeps the selected-pixel geometry overlay and central-ray indicator aligned with the selected SRM module.
