import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from steppo.utils.device import replicate_tree, shard_array

hidden = 2048
n_layers = 4
batch_per_device = 512
n_iters = 20

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, n_layers)
params = [(jax.random.normal(k, (hidden, hidden)) * 0.02, jnp.zeros((hidden,))) for k in keys]


def loss_fn(params, x, y):
    h = x
    for w, b in params:
        h = jnp.tanh(h @ w + b)
    return jnp.mean((h - y) ** 2)


def single_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)
    return new_params, loss


@jax.jit
def scan_steps(params, x, y):
    def body(params, _):
        new_params, loss = single_step(params, x, y)
        return new_params, loss

    new_params, losses = jax.lax.scan(body, params, None, length=n_iters)
    return new_params, losses[-1]


def benchmark(mesh):
    global_batch = batch_per_device * len(mesh.devices)
    x = shard_array(jax.random.normal(jax.random.PRNGKey(1), (global_batch, hidden)), mesh)
    y = shard_array(jax.random.normal(jax.random.PRNGKey(2), (global_batch, hidden)), mesh)
    p = replicate_tree(params, mesh)

    # Warmup / compile
    new_p, loss = scan_steps(p, x, y)
    jax.block_until_ready(loss)

    start = time.perf_counter()
    new_p, loss = scan_steps(p, x, y)
    jax.block_until_ready(loss)
    elapsed = time.perf_counter() - start

    return (global_batch * n_iters) / elapsed  # samples/sec


mesh_single = Mesh(jax.devices()[:1], ("data",))
mesh_multi = Mesh(jax.devices(), ("data",))

throughput_single = benchmark(mesh_single)
throughput_multi = benchmark(mesh_multi)
speedup = throughput_multi / throughput_single

print(f"1-device throughput:        {throughput_single:.1f} samples/sec")
print(f"{len(jax.devices())}-device throughput: {throughput_multi:.1f} samples/sec")
print(f"Speedup:                    {speedup:.2f}x")
