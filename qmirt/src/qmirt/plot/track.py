"""Track plotting helpers for qmirt_utility."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def create_plotly_go_track(
    track_df: Any, *, style: dict[str, Any] | None = None
) -> go.Scatter3d:
    """Create a Plotly track trace from a step table using post-step positions."""
    track_df = track_df.sort("CurrentStepNumber")
    post_step_positions = track_df.select(
        ["PostPosition_X", "PostPosition_Y", "PostPosition_Z"]
    ).to_numpy()
    return go.Scatter3d(
        x=post_step_positions[:, 0],
        y=post_step_positions[:, 1],
        z=post_step_positions[:, 2],
        mode="markers+lines",
        name=f"TrackID {track_df[0, 'TrackID']}, Process: {track_df[0, 'TrackCreatorProcess']}",
        **(style or {}),
    )


def create_plotly_go_track_with_steps(
    track_df: Any, *, style: dict[str, Any] | None = None
) -> go.Scatter3d:
    """Create a Plotly track trace with alternating pre-step and post-step positions."""
    track_df = track_df.sort("CurrentStepNumber")
    pre_step_positions = track_df.select(
        ["PrePosition_X", "PrePosition_Y", "PrePosition_Z"]
    ).to_numpy()
    post_step_positions = track_df.select(
        ["PostPosition_X", "PostPosition_Y", "PostPosition_Z"]
    ).to_numpy()
    points = pre_step_positions.repeat(2, axis=0)
    points[1::2] = post_step_positions
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers+lines",
        name=f"TrackID {track_df[0, 'TrackID']}, Process: {track_df[0, 'TrackCreatorProcess']}",
        **(style or {}),
    )


__all__ = ["create_plotly_go_track", "create_plotly_go_track_with_steps"]
