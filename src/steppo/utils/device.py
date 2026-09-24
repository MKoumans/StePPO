"""Device selection and sharding helpers for training entrypoints."""

import logging
import os
import re
import sys


def setup_devices(verbose: bool = True):
    """Configures the visible GPU/device context for JAX before it initializes.

    Looks for:
      - The '--gpus' command-line argument (e.g. `--gpus 1,2`)
      - The 'GPUS' environment variable
      - The 'CUDA_VISIBLE_DEVICES' environment variable

    Returns JAX's devices (empty if JAX failed to start). `verbose=False` keeps
    only warnings, for callers that report the devices themselves.
    """
    say = print if verbose else lambda *args, **kwargs: None
    gpus = None

    # 1. Parse from sys.argv (and remove it to prevent downstream parser errors)
    args_to_remove = []
    for idx, arg in enumerate(sys.argv):
        if arg == "--gpus":
            if idx + 1 < len(sys.argv):
                gpus = sys.argv[idx + 1]
                args_to_remove.extend([idx, idx + 1])
            else:
                args_to_remove.append(idx)
            break
        elif arg.startswith("--gpus="):
            gpus = arg.split("=", 1)[1]
            args_to_remove.append(idx)
            break

    # Remove arguments in reverse order to keep indices correct
    for idx in sorted(args_to_remove, reverse=True):
        sys.argv.pop(idx)

    # 2. Check environment variables if not provided via command line
    if gpus is None:
        gpus = os.environ.get("GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")

    # 3. Apply device settings
    if gpus is not None:
        gpus = gpus.strip().strip("'\"")
        # Handle CPU-only fallback
        if gpus.lower() in ("cpu", "none"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # Without this JAX still tries the CUDA plugin and reports it as a failure.
            os.environ["JAX_PLATFORMS"] = "cpu"
            say('[Device Setup] Forcing CPU-only execution (CUDA_VISIBLE_DEVICES="")')
        else:
            if os.environ.get("JAX_PLATFORMS") == "cpu":
                del os.environ["JAX_PLATFORMS"]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
            say(f"[Device Setup] Configured CUDA_VISIBLE_DEVICES={gpus}")
    else:
        say(
            "[Device Setup] No specific GPU device selection requested. Using default visible devices."
        )

    # 4. Import JAX and output available devices for clear feedback. JAX logs a
    # failing CUDA plugin as a full traceback and silently falls back to CPU;
    # capture that and replace it with an actionable message instead.
    plugin_errors = []
    log_filter = _CudaPluginErrorFilter(plugin_errors)
    xla_logger = logging.getLogger("jax._src.xla_bridge")
    xla_logger.addFilter(log_filter)
    try:
        import jax

        # JAX may already be imported (importing steppo does), so JAX_PLATFORMS
        # alone is too late; the config still applies until a backend starts.
        if os.environ.get("JAX_PLATFORMS") == "cpu":
            jax.config.update("jax_platforms", "cpu")
        devices = jax.devices()
    except Exception as e:
        print(f"[Device Setup] Warning: Failed to import JAX or retrieve devices: {e}")
        return []
    finally:
        xla_logger.removeFilter(log_filter)

    if plugin_errors:
        print(_cuda_failure_message(plugin_errors[0]), file=sys.stderr)
        if gpus is not None and gpus.lower() not in ("cpu", "none"):
            sys.exit(
                f"[Device Setup] GPU(s) {gpus} were requested but CUDA is unavailable; aborting."
            )
    say(f"[Device Setup] JAX successfully initialized. Available devices: {devices}")
    return devices


class _CudaPluginErrorFilter(logging.Filter):
    """Swallows JAX's CUDA plugin tracebacks, recording the exception instead."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def filter(self, record):
        msg = record.getMessage()
        if msg.startswith("Jax plugin configuration error"):
            exc = record.exc_info[1] if record.exc_info else None
            self.sink.append(exc if exc is not None else msg)
            return False
        # Follow-up "falling back to cpu" warning is redundant with our message.
        if self.sink and "Falling back to cpu" in msg:
            return False
        return True


# Known cuInit/driver failures -> likely cause and fix.
_CUDA_ERROR_HINTS = {
    "35": "The NVIDIA driver is older than the CUDA runtime JAX ships with. Update the GPU driver.",
    "100": "No CUDA-capable GPU is visible. In Docker, make sure the container was started with "
    "GPU access (`deploy.resources.reservations.devices` in docker-compose / `--gpus all`) "
    "and check CUDA_VISIBLE_DEVICES.",
    "303": "The CUDA driver library failed to initialise. On Windows/WSL2 (Docker Desktop) this "
    "usually means the GPU passthrough went stale, e.g. after sleep/hibernate or a driver update. "
    "Fix: stop the container, run `wsl --shutdown` (or restart Docker Desktop), and start it again; "
    "reboot Windows if that is not enough. Verify with `nvidia-smi` inside the container.",
    "803": "Driver/CUDA-compat mismatch: the container's CUDA libraries do not match the host "
    "driver. Update the host GPU driver or rebuild the image.",
    "804": "Forward-compatibility is not supported on this GPU; update the host GPU driver.",
    "999": "The GPU is in an unknown state. Reboot the machine (or `wsl --shutdown` on Windows).",
}


def _cuda_failure_message(error):
    """Builds a readable explanation for a failed JAX CUDA plugin initialisation."""
    detail = str(error).strip().splitlines()[-1] if str(error).strip() else repr(error)
    match = re.search(r"CUDA error (\d+)", detail)
    hint = _CUDA_ERROR_HINTS.get(match.group(1)) if match else None
    if hint is None:
        hint = (
            "Check that the NVIDIA driver works (`nvidia-smi`), that the container has GPU "
            "access, and that the driver supports CUDA 12."
        )
    return (
        "[Device Setup] WARNING: JAX could not initialise CUDA and is falling back to CPU.\n"
        f"  Cause: {detail}\n"
        f"  Likely fix: {hint}\n"
        "  To run on CPU intentionally, pass `--gpus cpu` (or set GPUS=cpu)."
    )


def get_device_mesh():
    """Initialises a 1D JAX device mesh for data-parallel training."""
    import jax
    from jax.sharding import Mesh

    devices = jax.devices()
    return Mesh(devices, ("data",))


def shard_array(x, mesh):
    """Shards a JAX array along its batch dimension (axis 0) across the device mesh.

    If the mesh contains 1 device or is None, does nothing.
    """
    if mesh is None or len(mesh.devices) <= 1:
        return x

    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    # Ensure x is a JAX array
    x = jax.device_put(x)
    ndim = x.ndim
    if ndim == 0:
        # Cannot shard a scalar, replicate it
        sharding = NamedSharding(mesh, P())
    else:
        # Shard the first dimension (batch) and replicate the rest
        sharding = NamedSharding(mesh, P("data", *([None] * (ndim - 1))))
    return jax.device_put(x, sharding)


def shard_tree(tree, mesh):
    """Shard every array leaf of a pytree along its leading dimension."""
    if mesh is None or len(mesh.devices) <= 1:
        return tree

    import jax

    return jax.tree.map(lambda x: shard_array(x, mesh), tree)


def replicate_array(x, mesh):
    """Replicates a JAX array across all devices in the device mesh.

    If the mesh contains 1 device or is None, does nothing.
    """
    if mesh is None or len(mesh.devices) <= 1:
        return x

    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    x = jax.device_put(x)
    sharding = NamedSharding(mesh, P())
    return jax.device_put(x, sharding)


def replicate_tree(tree, mesh):
    """Replicate every array leaf of a pytree across the device mesh."""
    if mesh is None or len(mesh.devices) <= 1:
        return tree

    import jax

    return jax.tree.map(lambda x: replicate_array(x, mesh), tree)
