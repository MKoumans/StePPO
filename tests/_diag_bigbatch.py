import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from steppo.utils.device import replicate_tree, shard_array

hidden = 2048
n_layers = 4
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


def benchmark(mesh, batch_per_device):
    global_batch = batch_per_device * len(mesh.devices)
    x = shard_array(jax.random.normal(jax.random.PRNGKey(1), (global_batch, hidden)), mesh)
    y = shard_array(jax.random.normal(jax.random.PRNGKey(2), (global_batch, hidden)), mesh)
    p = replicate_tree(params, mesh)

    loss, p = step(p, x, y)
    jax.block_until_ready(loss)

    start = time.perf_counter()
    for _ in range(n_iters):
        loss, p = step(p, x, y)
    jax.block_until_ready(loss)
    elapsed = time.perf_counter() - start

    return (global_batch * n_iters) / elapsed


mesh_single = Mesh(jax.devices()[:1], ("data",))
mesh_multi = Mesh(jax.devices(), ("data",))

for batch_per_device in (512, 4096):
    t1 = benchmark(mesh_single, batch_per_device)
    tn = benchmark(mesh_multi, batch_per_device)
    print(
        f"batch_per_device={batch_per_device}: 1-dev={t1:.1f} samples/s, "
        f"{len(jax.devices())}-dev={tn:.1f} samples/s, speedup={tn / t1:.2f}x"
    )
