"""Formatting helper utilities for qmirt_utility."""

from __future__ import annotations


def print_list_aligned(
    items: list,
    *,
    vline: bool = True,
    hline: bool = True,
    box: bool = True,
    fixed_width: int | None = None,
    min_width: int | None = None,
):
    """Print a list of rows with aligned columns and optional box drawing."""
    assert all(isinstance(item, list) for item in items), "Expected a list of lists!"

    num_columns = max(len(row) for row in items)
    column_widths = [0] * num_columns
    for row in items:
        for i, item in enumerate(row):
            column_widths[i] = max(column_widths[i], len(str(item)))

    if fixed_width is not None:
        column_widths = [fixed_width] * num_columns
    if min_width is not None:
        column_widths = [max(width, min_width) for width in column_widths]

    len_line = sum((w + 2) for w in column_widths)
    inner_width = len_line + (1 if vline else 0)

    def _print_top():
        if vline:
            print("┌" + "─" * inner_width + "┐")
        else:
            print("─" * inner_width)

    def _print_bottom():
        if vline:
            print("└" + "─" * inner_width + "┘")
        else:
            print("─" * inner_width)

    if box:
        _print_top()

    for row in items:
        line = ""
        if not vline:
            for i, item in enumerate(row):
                line += str(item).ljust(column_widths[i]) + "  "
        else:
            for i, item in enumerate(row[:-1]):
                line += str(item).ljust(column_widths[i]) + " │ "
            if row:
                line += str(row[-1]).ljust(column_widths[len(row) - 1]) + " "

        if box:
            line = "│ " + line + "│"
        print(line)

    if hline:
        _print_bottom()


__all__ = ["print_list_aligned"]
