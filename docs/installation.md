# Installation

## Choosing a setup

| Your machine | CPU only | With an NVIDIA GPU |
| --- | --- | --- |
| Linux | [local install](#local-install) | the same local install (or [Docker](#docker)) |
| Windows | [local install](#local-install): `uv run steppo ...` | the [dev container](#windows-docker-desktop-or-podman): `steppo ...` |
| macOS (Apple Silicon) | [local install](#macos) | not available |

Both Windows routes work from the same checkout. `uv run steppo solve ...` runs
natively on the CPU; `steppo solve ... --gpus 0` (the `steppo.cmd` wrapper)
runs in the container, where JAX can use the GPU. A single `steppo solve` is
faster on the CPU, so the GPU pays off mainly for training and batched
evaluation.

## Prerequisites

- Python 3.11+
- CUDA 12 for GPU acceleration, on Linux only (there JAX is installed with the
  `cuda12` extra; on macOS and Windows `uv sync` installs CPU-only JAX)
- [uv](https://docs.astral.sh/uv/)

## Local install

```bash
uv sync
uv sync --extra logging   # optional: adds Weights & Biases
```

`uv sync` creates `.venv/` from the pinned versions in `uv.lock`. Prefix
commands with `uv run` (e.g. `uv run pytest`), or activate the environment with
`source .venv/bin/activate`. After changing dependencies in `pyproject.toml`,
run `uv lock` and commit the updated `uv.lock`.

`PYTHONPATH=.` is required for the entrypoints under `research/`, which is not
part of the importable package — `steppo` itself is picked up via the editable
install `uv sync` creates from `src/steppo/`.

## macOS

On Apple Silicon (M1 or later), install natively; JAX runs on the CPU:

```bash
uv sync
uv run steppo solve van_der_pol --xi 50
```

There is no GPU route on a Mac: JAX has no maintained Apple GPU backend, and
containers cannot reach the Apple GPU. Training works but is CPU-only, so large
runs belong on a Linux GPU machine (or a Windows machine with the dev
container). Intel Macs are not supported: JAX publishes no wheels for them.

## Docker

`docker/Dockerfile` + `docker/docker-compose.yml` build a CUDA 12 container with
uv-managed dependencies, bind-mounting the
repository at `/workspace`. The dependencies live in `/opt/venv`, which is on
`PATH`; `uv run` inside the container reuses that environment.

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec app bash
```

The compose service reserves all NVIDIA devices and honours `CUDA_VISIBLE_DEVICES`
and `GPUS` from the host environment.

The DeepONet baseline needs PyTorch/DeepXDE, whose CUDA libraries conflict with
`jax[cuda12]`, so it has its own image:

```bash
docker compose -f docker/docker-compose.deeponet.yml up -d
docker compose -f docker/docker-compose.deeponet.yml exec deeponet bash
```

See [`research/baselines/deeponet/`](../research/baselines/deeponet/README.md).

## Windows (Docker Desktop or Podman)

To use the GPU on Windows, run the project in the dev container. Docker Desktop
and Podman both run it in a WSL 2 VM that sees the Windows NVIDIA driver. The commands below are for PowerShell, run from the
repository root.

### Prerequisites

- Windows 10 21H2+ or Windows 11 with WSL 2 (`wsl --install`, then reboot).
- A current NVIDIA driver for **Windows**. Do not install a Linux NVIDIA driver
  inside WSL; the Windows driver is shared with the VM.
- One container engine:
  - **Docker Desktop**, with the WSL 2 backend (the default). GPU support
    works out of the box.
  - **Podman**: `winget install -e --id RedHat.Podman`, or Podman Desktop.
    `podman compose` hands off to an external compose provider; Podman Desktop
    installs `docker-compose`, which is the tested provider.

`.gitattributes` checks out every text file with LF line endings, so the shell
scripts run in the container even with `core.autocrlf=true`.

### One-time Podman setup

Create and start a rootful machine (for an existing machine:
`podman machine stop; podman machine set --rootful; podman machine start`):

```powershell
podman machine init --rootful
podman machine start
```

Then give containers access to the GPU, from the repository root in cmd:

```bat
steppo setup-gpu
```

This installs the NVIDIA Container Toolkit in the machine, generates its CDI
spec (the GPU becomes the device `nvidia.com/gpu=all`) and checks that a
container sees the GPU; it ends by printing `GPU 0: NVIDIA ...`. Run it again
after every Windows NVIDIA driver update, since the spec records driver file
paths. The same steps by hand:

```powershell
podman machine ssh "curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo"
podman machine ssh "sudo dnf install -y nvidia-container-toolkit"
podman machine ssh "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu:22.04 nvidia-smi -L
```

Docker Desktop needs no GPU setup.

### Everyday use: the `steppo.cmd` wrapper

`steppo.cmd` in the repository root runs commands inside the dev container. It
starts the Podman machine and the container when they are not running; the
first start builds the image, which downloads several GB and can take 10
minutes or more. It uses Podman when installed, otherwise Docker; set
`STEPPO_ENGINE=docker` (or `podman`) to choose.

From the repository root in cmd (in PowerShell, type `.\steppo`):

```bat
steppo solve van_der_pol --xi 50     :: the quickstart: StePPO vs PID
steppo jupyter                       :: JupyterLab for examples\quickstart.ipynb
steppo shell                         :: bash in the container (uv run pytest, training, ...)
steppo stop                          :: stop the container
steppo                               :: list the commands
```

- File paths in arguments are relative to the repository root, which is
  mounted at `/workspace`; files the container writes appear in your checkout.
- `steppo jupyter` prints the link to open, including its access token. Under
  Podman the link uses the VM's address (such as `http://172.22.168.172:8888`)
  rather than `127.0.0.1`: a rootful Podman machine forwards ports with NAT
  rules, which WSL does not relay to Windows `localhost`.
- `steppo solve` runs on the CPU by default, because a single solve is several
  times faster there than on a GPU. To use a GPU anyway, pass `--gpus 0` or
  `set GPUS=0` first.
- The released models are part of the image, so `steppo solve` works offline;
  `--refresh` downloads them from Hugging Face again. Compiled programs are
  cached in the `steppo_cache` volume, so repeat runs start faster.

### Manual commands

The wrapper runs these compose commands. Under Podman, add the override
`docker/docker-compose.podman.yml`: Podman ignores compose's GPU reservation, and
the override requests the CDI device instead.

| Step | Docker Desktop | Podman |
| --- | --- | --- |
| Build and start | `docker compose -f docker/docker-compose.yml up -d --build` | `podman compose -f docker/docker-compose.yml -f docker/docker-compose.podman.yml up -d --build` |
| Open a shell | `docker compose -f docker/docker-compose.yml exec app bash` | `podman compose -f docker/docker-compose.yml -f docker/docker-compose.podman.yml exec app bash` |
| Stop | `docker compose -f docker/docker-compose.yml down` | `podman compose -f docker/docker-compose.yml -f docker/docker-compose.podman.yml down` |

After the first build, `up -d` without `--build` starts the container in
seconds; rebuild only when `pyproject.toml`, `uv.lock` or the Dockerfile
changes. Without compose, build and run the image directly:

```powershell
# Docker Desktop
docker build -f docker/Dockerfile -t steppo-dev .
docker run -it --rm --gpus all --shm-size 4g -p 8888:8888 -v "${PWD}:/workspace" -w /workspace steppo-dev bash

# Podman
podman build -f docker/Dockerfile -t steppo-dev .
podman run -it --rm --device nvidia.com/gpu=all --shm-size 4g -p 8888:8888 -v "${PWD}:/workspace" -w /workspace steppo-dev bash
```

Inside the container, check that JAX finds the GPU, then work as on Linux:

```bash
uv run python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
uv run pytest
uv run --with jupyterlab jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

The DeepONet container follows the same compose pattern with
`-f docker/docker-compose.deeponet.yml` and service `deeponet`. It has no
Podman override yet, so under Podman it runs without the GPU; use
`podman run --device nvidia.com/gpu=all ...` for GPU runs.

### Troubleshooting

- **`unauthorized: incorrect username or password` while pulling
  `nvidia/cuda`.** A stale Docker Hub login is stored in Docker Desktop's
  credential store, which `podman compose` (through `docker-compose`) also
  uses. Run `docker logout`, or pull the base image once without it:
  `podman pull docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`.
- **JAX prints `cuInit(0) failed` and falls back to `CpuDevice`.** The
  container has no GPU. Under Podman, check that you used the wrapper or the
  `docker-compose.podman.yml` override, re-run `steppo setup-gpu` (needed after
  driver updates), and check that `podman machine ssh "nvidia-ctk cdi list"`
  lists `nvidia.com/gpu=all`.
- **`http://127.0.0.1:8888` does not load under Podman.** Use the link
  `steppo jupyter` prints, which points at the Podman VM's address.
- **`CUDA_ERROR_OUT_OF_MEMORY` lines on a laptop GPU.** XLA retries smaller
  allocations; if the command still finishes, the messages are harmless.

## HPC (Slurm / Apptainer)

`scripts/setup/create-apptainer.sh` builds the Apptainer image used on Slurm
clusters. Every launcher in `scripts/`
auto-detects its execution mode: if `sbatch` is on `PATH` the work is submitted
as Slurm jobs dispatching into the Apptainer image, otherwise it runs directly on
the current machine. See [`scripts/README.md`](../scripts/README.md).

## GPU and JAX notes

- `src/steppo/utils/device.py`'s `setup_devices()` sets `GPUS`/`CUDA_VISIBLE_DEVICES`
  and disables XLA memory preallocation. It must run **before** JAX is imported,
  which is why every entrypoint calls it on the first two lines of the module.
- `jax_enable_x64` is enabled globally, with no reset, by `apply_precision()` and
  by `ODEEnv.__init__` when `env.precision: float64`. It therefore leaks across
  processes that build several configs in sequence; the test suite restores it
  after every test (see [development.md](development.md)).
- Solver arithmetic precision (`env.precision`) and neural-network dtype
  (`model_precision`) are configured independently. Step counts are not
  comparable across solver precisions.
