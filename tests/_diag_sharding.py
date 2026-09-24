import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from steppo.utils.device import replicate_tree, shard_array

mesh = Mesh(jax.devices(), ("data",))
x = shard_array(jnp.zeros((4096, 2048)), mesh)
print("x sharding:", x.sharding)
print("per-shard shapes:", set(s.data.shape for s in x.addressable_shards))

hidden = 2048
params = [(jnp.zeros((hidden, hidden)), jnp.zeros((hidden,))) for _ in range(4)]
p = replicate_tree(params, mesh)
print("param sharding:", p[0][0].sharding)
print("param per-shard shapes:", set(s.data.shape for s in p[0][0].addressable_shards))


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


y = shard_array(jnp.zeros((4096, 2048)), mesh)
compiled = step.lower(p, x, y).compile()
txt = compiled.as_text()
print("num all-reduce ops:", txt.count("all-reduce"))
print("num all-gather ops:", txt.count("all-gather"))
print("num collective-permute ops:", txt.count("collective-permute"))

# Also check the loss output sharding vs grads sharding
loss, new_p = step(p, x, y)
print("loss sharding:", loss.sharding)
print("grad/new_param sharding:", new_p[0][0].sharding)
print("grad per-shard shapes:", set(s.data.shape for s in new_p[0][0].addressable_shards))
