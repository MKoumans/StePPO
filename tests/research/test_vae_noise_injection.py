"""Deterministic checks for the analysis' temporal noise intervention."""

import numpy as np

from research.ode.post_run_analysis.vae_noise_injection import (
    encode_clean_and_distorted,
    spike_step_indices,
)


def test_spike_indices_choose_first_step_at_or_after_each_fraction():
    state_t = np.array([0.0, 0.1, 0.4, 0.9, 1.0])

    assert spike_step_indices(state_t, 1.0, [0.0, 0.2, 0.4, 0.95, 2.0]) == [0, 2, 2, 4, 4]


def test_noise_ratios_scale_a_fixed_direction_only_at_spike_steps():
    class Encoder:
        def __init__(self):
            self.inputs = []

        def encode_trajectory(self, _actions, obs, _rewards):
            obs = np.asarray(obs)
            self.inputs.append(obs.copy())
            # Use observations as means and an out-of-range log variance to
            # verify the public sigma output remains finite after clipping.
            return obs[:, :2], np.full((len(obs), 2), 40.0)

    encoder = Encoder()
    vae = type("FakeVAE", (), {"encoder": encoder})()
    obs = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [2.0, 2.0, 6.0],
            [4.0, 5.0, 8.0],
        ]
    )
    original = obs.copy()

    normal, distorted = encode_clean_and_distorted(
        vae,
        obs,
        action=np.zeros((3, 1)),
        reward=np.zeros((3, 1)),
        spike_idxs=[1, 3],
        noise_ratios=[0.0, 0.5, 1.0],
        rng=np.random.default_rng(123),
    )

    assert len(encoder.inputs) == 3  # clean encoding and nonzero ratios only
    assert np.array_equal(obs, original)
    assert np.array_equal(distorted[0.0]["mu"], normal["mu"])
    assert np.array_equal(distorted[0.0]["sigma"], normal["sigma"])
    assert np.allclose(normal["sigma"], np.exp(5.0))

    half_noise = encoder.inputs[1] - original
    full_noise = encoder.inputs[2] - original
    assert np.allclose(half_noise[[0, 2]], 0.0)
    assert np.allclose(full_noise[[0, 2]], 0.0)
    assert np.any(half_noise[[1, 3]] != 0.0)
    assert np.allclose(full_noise, 2.0 * half_noise)
    assert np.array_equal(distorted[0.5]["mu"], encoder.inputs[1][:, :2])
    assert np.array_equal(distorted[1.0]["mu"], encoder.inputs[2][:, :2])
