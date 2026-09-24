"""Shape and masking invariants shared by post-run analysis plots."""

import jax.numpy as jnp
import numpy as np

from research.ode.post_run_analysis.analysis_common import col_edges, freeze_inactive


def test_freeze_inactive_preserves_completed_trajectories():
    active = jnp.array([True, False, True])
    new = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    old = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert np.array_equal(
        freeze_inactive(active, new, old),
        [
            [10.0, 11.0],
            [3.0, 4.0],
            [30.0, 31.0],
        ],
    )


def test_col_edges_extrapolate_half_cells_per_row():
    centers = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]])

    edges = col_edges(centers)

    assert edges.shape == (2, 4)
    assert np.array_equal(edges, [[0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 6.0, 10.0]])
