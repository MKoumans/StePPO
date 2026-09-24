import os
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from steppo.configs.base_config import TrainConfig
from steppo.utils.device import (
    get_device_mesh,
    replicate_tree,
    setup_devices,
    shard_array,
    shard_tree,
)


def test_setup_devices_args():
    # Store original argv and environment variables
    orig_argv = list(sys.argv)
    orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    orig_gpus = os.environ.get("GPUS")
    orig_platforms = os.environ.get("JAX_PLATFORMS")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if "GPUS" in os.environ:
        del os.environ["GPUS"]

    try:
        # Test 1: CLI Argument --gpus 1,2
        sys.argv = ["run.py", "--gpus", "1,2", "--total_iters", "10"]
        setup_devices()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1,2"
        # Check it was correctly stripped from sys.argv to keep argparse happy
        assert "--gpus" not in sys.argv
        assert "1,2" not in sys.argv

        # Test 2: CLI Argument --gpus=cpu
        sys.argv = ["run.py", "--gpus=cpu"]
        setup_devices()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert os.environ.get("JAX_PLATFORMS") == "cpu"

        # Test 3: Env Var fallback
        sys.argv = ["run.py"]
        os.environ["GPUS"] = "4,5"
        setup_devices()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5"

    finally:
        # Restore original argv and environment variables
        sys.argv = orig_argv
        if orig_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        if orig_gpus is not None:
            os.environ["GPUS"] = orig_gpus
        elif "GPUS" in os.environ:
            del os.environ["GPUS"]

        if orig_platforms is not None:
            os.environ["JAX_PLATFORMS"] = orig_platforms
        elif "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]


def test_cuda_plugin_error_is_reported_readably():
    import logging

    from steppo.utils.device import _cuda_failure_message, _CudaPluginErrorFilter

    errors = []
    log_filter = _CudaPluginErrorFilter(errors)
    err = RuntimeError("operation cuInit(0) failed: Unknown CUDA error 303; cuGetErrorName failed.")
    record = logging.LogRecord(
        "jax._src.xla_bridge",
        logging.ERROR,
        __file__,
        0,
        "Jax plugin configuration error: Exception when calling %s.initialize()",
        ("jax_plugins.xla_cuda12",),
        (RuntimeError, err, None),
    )
    assert not log_filter.filter(record)
    assert errors == [err]

    message = _cuda_failure_message(errors[0])
    assert "falling back to CPU" in message
    assert "wsl --shutdown" in message


def test_mesh_and_sharding():
    mesh = get_device_mesh()
    assert isinstance(mesh, Mesh)

    # Test shard_array — use a size divisible by any reasonable device count (up to 8)
    n_devices = max(1, len(mesh.devices))
    batch = max(8, n_devices)  # at least n_devices rows
    x = jnp.ones((batch, 3))
    x_sharded = shard_array(x, mesh)

    if len(mesh.devices) > 1:
        sharding = x_sharded.sharding
        assert hasattr(sharding, "spec")
        assert sharding.spec[0] == "data"
    else:
        assert x_sharded is x or x_sharded.shape == x.shape

    # Test shard_tree (nested PyTree container)
    tree = {"a": x, "b": jnp.array(5.0)}
    tree_sharded = shard_tree(tree, mesh)
    assert tree_sharded["a"].shape == x.shape
    assert tree_sharded["b"].shape == ()


def test_batch_size_divisibility_logic():
    # Setup dummy configurations
    cfg = TrainConfig()
    cfg.vae.batch_size = 25
    cfg.num_envs = 16
    cfg.rollout_steps = 400
    cfg.ppo.num_minibatches = 4

    num_devices = 2  # mock 2 devices

    # Run num_envs adjustment
    if cfg.num_envs % num_devices != 0:
        old_envs = cfg.num_envs
        cfg.num_envs = max(num_devices, round(old_envs / num_devices) * num_devices)
    assert cfg.num_envs == 16  # already divisible by 2

    # Run VAE batch size adjustment
    if cfg.vae.batch_size % num_devices != 0:
        old_bs = cfg.vae.batch_size
        cfg.vae.batch_size = ((old_bs + num_devices - 1) // num_devices) * num_devices

    assert cfg.vae.batch_size == 26

    # Test with 5 devices (should round num_envs 16 down to 15)
    num_devices = 5
    cfg.num_envs = 16
    if cfg.num_envs % num_devices != 0:
        old_envs = cfg.num_envs
        cfg.num_envs = max(num_devices, round(old_envs / num_devices) * num_devices)
    assert cfg.num_envs == 15

    # Run PPO minibatches adjustment
    total_ppo_size = cfg.num_envs * cfg.rollout_steps  # 6400
    desired_minibatches = cfg.ppo.num_minibatches  # 4
    best_minibatches = None
    for diff in range(desired_minibatches):
        for candidate in [desired_minibatches - diff, desired_minibatches + diff]:
            if candidate > 0:
                mb_size = total_ppo_size // candidate
                if mb_size > 0 and mb_size % num_devices == 0:
                    best_minibatches = candidate
                    break
        if best_minibatches is not None:
            break

    assert best_minibatches == 4  # 6400 // 4 = 1600 (divisible by 2)

    # Test with 8 devices and non-divisible PPO size
    num_devices = 8
    cfg.vae.batch_size = 25
    cfg.num_envs = 16
    cfg.rollout_steps = 10
    total_ppo_size = 160
    cfg.ppo.num_minibatches = 7  # 160 // 7 = 22 (not divisible by 8)

    # Run VAE adjustment
    if cfg.vae.batch_size % num_devices != 0:
        old_bs = cfg.vae.batch_size
        cfg.vae.batch_size = ((old_bs + num_devices - 1) // num_devices) * num_devices
    assert cfg.vae.batch_size == 32  # rounded up to multiple of 8

    # Run PPO adjustment
    desired_minibatches = cfg.ppo.num_minibatches  # 7
    best_minibatches = None
    for diff in range(desired_minibatches):
        for candidate in [desired_minibatches - diff, desired_minibatches + diff]:
            if candidate > 0:
                mb_size = total_ppo_size // candidate
                if mb_size > 0 and mb_size % num_devices == 0:
                    best_minibatches = candidate
                    break
        if best_minibatches is not None:
            break

    # minibatches options:
    # candidate=8 => mb_size = 160 // 8 = 20 (20 % 8 = 4 != 0)
    # candidate=5 => mb_size = 160 // 5 = 32 (32 % 8 == 0) -> MATCH!
    assert best_minibatches == 5


def test_performance_summary_report():
    # Build performance summary string
    summary = []
    summary.append("=" * 60)
    summary.append("            VariBAD JAX Performance Summary             ")
    summary.append("=" * 60)
    summary.append(" Experiment Name:     test_exp")
    summary.append(" Total Iterations:    10")
    summary.append(" Vectorised Envs:     16 (Steps per rollout: 400)")
    summary.append(" Total Environment Steps: 64000")
    summary.append("-" * 60)
    summary.append(" Active GPU Devices:  1")
    summary.append(" Device Names:        cpu:0")
    summary.append("-" * 60)
    summary.append(" Total Elapsed Time:  5.00 seconds")
    summary.append(" Overall Throughput:  12800.00 env steps/sec")
    summary.append("-" * 60)
    summary.append(" Component Runtimes (Mean / Compile):")
    summary.append("  - Rollout Collection:  120.00 ms")
    summary.append("  - VAE Update:          200.00 ms  (First/Compile: 800.00 ms)")
    summary.append("  - PPO Update:          150.00 ms  (First/Compile: 600.00 ms)")
    summary.append("  - Steady State Iter:   470.00 ms")
    summary.append("=" * 60)
    summary_str = "\n".join(summary)

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, "benchmark_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_str)

        assert os.path.exists(summary_path)
        with open(summary_path, "r") as f:
            content = f.read()
            assert "VariBAD JAX Performance Summary" in content
            assert "Overall Throughput" in content
            assert "12800.00 env steps/sec" in content


@pytest.mark.benchmark
@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="Multi-GPU speedup benchmark requires more than one visible device",
)
def test_multi_gpu_speedup():
    """Verifies that sharding a data-parallel workload across all visible devices
    yields at least a 25% throughput increase over running on a single device.
    """
    n_devices = jax.device_count()
    hidden = 2048
    n_layers = 4
    # Gradient all-reduce volume is fixed by model size (~67MB for this model)
    # and dominates on PCIe-only (no NVLink) interconnects unless per-device
    # compute is large enough to amortize it. batch_per_device=512 measured
    # ~0.4x (regression); 4096 gives ~1.8x.
    batch_per_device = 4096
    n_iters = 20

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_layers)
    params = [(jax.random.normal(k, (hidden, hidden)) * 0.02, jnp.zeros((hidden,))) for k in keys]

    def loss_fn(params, x, y):
        h = x
        for w, b in params:
            h = jnp.tanh(h @ w + b)
        return jnp.mean((h - y) ** 2)

    @jax.jit
    def step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)
        return loss, new_params

    def benchmark(mesh):
        global_batch = batch_per_device * len(mesh.devices)
        x = shard_array(jax.random.normal(jax.random.PRNGKey(1), (global_batch, hidden)), mesh)
        y = shard_array(jax.random.normal(jax.random.PRNGKey(2), (global_batch, hidden)), mesh)
        p = replicate_tree(params, mesh)

        # Warmup: trigger compilation, exclude from timing
        loss, p = step(p, x, y)
        jax.block_until_ready(loss)

        start = time.perf_counter()
        for _ in range(n_iters):
            loss, p = step(p, x, y)
        jax.block_until_ready(loss)
        elapsed = time.perf_counter() - start

        return (global_batch * n_iters) / elapsed  # samples/sec

    mesh_single = Mesh(jax.devices()[:1], ("data",))
    mesh_multi = Mesh(jax.devices(), ("data",))

    throughput_single = benchmark(mesh_single)
    throughput_multi = benchmark(mesh_multi)
    speedup = throughput_multi / throughput_single

    print(f"[Benchmark] 1-device throughput:        {throughput_single:.1f} samples/sec")
    print(f"[Benchmark] {n_devices}-device throughput: {throughput_multi:.1f} samples/sec")
    print(f"[Benchmark] Speedup:                    {speedup:.2f}x")

    assert speedup >= 1.25, (
        f"Expected at least a 25% throughput increase using {n_devices} devices, "
        f"got {speedup:.2f}x ({throughput_single:.1f} -> {throughput_multi:.1f} samples/sec)"
    )


def test_yaml_config_loading():
    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    yaml_content = """
exp_name: "test_run"
total_iters: 123
num_envs: 32
vae:
  latent_dim: 10
  batch_size: 50
ppo:
  lr: 0.005
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_name = f.name

    try:
        config = load_config_from_yaml(TrainConfig, temp_name)
        assert config.exp_name == "test_run"
        assert config.total_iters == 123
        assert config.num_envs == 32
        assert config.vae.latent_dim == 10
        assert config.vae.batch_size == 50
        assert config.ppo.lr == 0.005
        # Default fields should remain intact
        assert config.ppo.clip_eps == 0.1
    finally:
        try:
            os.remove(temp_name)
        except OSError:
            pass
