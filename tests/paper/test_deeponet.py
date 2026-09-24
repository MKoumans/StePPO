"""NumPy DeepONet (paper/utils.py) against an unvectorized per-element reference."""

import numpy as np
import pytest

from paper.utils import DeepONet


def _linear(rng, n_in, n_out):
    return rng.standard_normal((n_out, n_in)).astype(np.float32), rng.standard_normal(n_out).astype(
        np.float32
    )


def _add_fnn(params, rng, prefix, sizes):
    for j, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        params[f"{prefix}{j}.weight"], params[f"{prefix}{j}.bias"] = _linear(rng, n_in, n_out)


def _add_msffn(params, rng, prefix, trunk_layers, n_sigmas):
    in_dim, fourier_dim, *hidden, out_dim = trunk_layers
    for i in range(n_sigmas):
        params[f"{prefix}b{i}"] = rng.standard_normal((in_dim, fourier_dim // 2)).astype(np.float32)
    prev = fourier_dim
    for j, size in enumerate(hidden):
        params[f"{prefix}hidden.{j}.weight"], params[f"{prefix}hidden.{j}.bias"] = _linear(
            rng, prev, size
        )
        prev = size
    params[f"{prefix}out.weight"], params[f"{prefix}out.bias"] = _linear(
        rng, prev * n_sigmas, out_dim
    )


def _make_params(
    num_outputs, msffn, rng, branch_layers=(1, 6, 5), trunk_layers=(1, 8, 7, 5), n_sigmas=2
):
    params = {}
    for k in range(num_outputs):
        tag = f".{k}" if num_outputs > 1 else ""
        _add_fnn(params, rng, f"branch{tag}.linears.", list(branch_layers))
        if msffn:
            _add_msffn(params, rng, f"trunk{tag}.", list(trunk_layers), n_sigmas)
        else:
            _add_fnn(params, rng, f"trunk{tag}.linears.", list(trunk_layers))
        params[f"b.{k}"] = np.array(rng.standard_normal(), dtype=np.float32)
    return params


def _relu(x):
    return x if x > 0 else 0.0


def _ref_fnn(params, prefix, x, last_activation=False):
    n_layers = sum(k.startswith(prefix) and k.endswith(".weight") for k in params)
    for j in range(n_layers):
        w, b = params[f"{prefix}{j}.weight"], params[f"{prefix}{j}.bias"]
        x = [sum(w[o, i] * x[i] for i in range(len(x))) + b[o] for o in range(w.shape[0])]
        if j < n_layers - 1 or last_activation:
            x = [_relu(v) for v in x]
    return x


def _ref_msffn(params, prefix, x):
    feats = []
    i = 0
    while f"{prefix}b{i}" in params:
        b = params[f"{prefix}b{i}"]
        proj = [sum(x[d] * b[d, c] for d in range(len(x))) for c in range(b.shape[1])]
        y = [np.cos(p) for p in proj] + [np.sin(p) for p in proj]
        y = _ref_fnn(params, f"{prefix}hidden.", y, last_activation=True)
        feats.extend(y)
        i += 1
    w, bias = params[f"{prefix}out.weight"], params[f"{prefix}out.bias"]
    return [sum(w[o, c] * feats[c] for c in range(len(feats))) + bias[o] for o in range(w.shape[0])]


def _reference(params, num_outputs, msffn, xi, t):
    out = np.empty((len(xi), len(t), num_outputs))
    for k in range(num_outputs):
        tag = f".{k}" if num_outputs > 1 else ""
        for a, x in enumerate(xi):
            branch = _ref_fnn(params, f"branch{tag}.linears.", [x])
            for c, s in enumerate(t):
                trunk = (
                    _ref_msffn(params, f"trunk{tag}.", [s])
                    if msffn
                    else _ref_fnn(params, f"trunk{tag}.linears.", [s])
                )
                trunk = [_relu(v) for v in trunk]
                out[a, c, k] = sum(bv * tv for bv, tv in zip(branch, trunk)) + params[f"b.{k}"]
    return out


@pytest.mark.parametrize("num_outputs", [1, 2])
@pytest.mark.parametrize("msffn", [False, True])
def test_forward_matches_unvectorized_reference(num_outputs, msffn):
    rng = np.random.default_rng(0)
    params = _make_params(num_outputs, msffn, rng)
    xi = np.array([0.5, 2.0, 7.0], dtype=np.float32)
    t = np.array([0.0, 0.3, 1.1, 4.0], dtype=np.float32)

    model = DeepONet(params)
    assert model.num_outputs == num_outputs
    assert model.is_msffn_trunk() is msffn
    np.testing.assert_allclose(
        model(xi, t), _reference(params, num_outputs, msffn, xi, t), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_msffn_spec_recovers_trunk_layers(num_outputs):
    rng = np.random.default_rng(1)
    params = _make_params(num_outputs, True, rng, trunk_layers=(1, 10, 9, 8, 4), n_sigmas=3)
    assert DeepONet(params).msffn_spec() == (3, [1, 10, 9, 8, 4])


def test_plain_trunk_layer_sizes_unchanged():
    rng = np.random.default_rng(2)
    params = _make_params(2, False, rng, branch_layers=(1, 6, 5), trunk_layers=(1, 8, 7, 5))
    model = DeepONet(params)
    assert model.layer_sizes("branch") == [1, 6, 5]
    assert model.layer_sizes("trunk") == [1, 8, 7, 5]
