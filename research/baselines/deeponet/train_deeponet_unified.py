"""Train DeepONet baselines for scalar_decay, van_der_pol and brusselator on two
task domains, with one shared architecture and training setup.

--domain complete  trains on the system's test_bins.
--domain binned    trains on a gapped subset of test_bins (_BINNED_TRAIN_BINS),
                   to measure interpolation and extrapolation.
Evaluation always uses env.test_bins from the system's default config.

Run in the DeepONet (PyTorch/DeepXDE) container:
    python train_deeponet_unified.py --system van_der_pol --domain complete
    python train_deeponet_unified.py --profile retrain --all --domain complete

The retrain profile (RETRAIN_SETTINGS) is the recipe of the paper's DeepONets:
8192 train / 8192 test trajectories, mini-batches of 256, 30000 iterations and a
multi-scale Fourier feature trunk (sigma = 1, 10). It writes a JSON manifest next
to each loss history.
``--dataset-source cache`` trains on the cached reference trajectories,
interpolated onto a common time grid.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import yaml
from dataset_generation import (
    generate_trajectories,
    sample_binned_values,
    sample_tasks,
    scalar_decay_trajectories_on_device,
)

HERE = os.path.dirname(__file__)
CONFIGS_ROOT = os.path.join(HERE, "..", "..", "..", "configs", "envs", "ode")

SYSTEMS = ("scalar_decay", "van_der_pol", "brusselator")
DOMAINS = ("complete", "binned")
NUM_OUTPUTS = {"scalar_decay": 1, "van_der_pol": 2, "brusselator": 2}

_BINNED_TRAIN_BINS = {
    "scalar_decay": [[5.0, 10.0], [20.0, 35.0], [60.0, 100.0]],  # gapped bins
    "van_der_pol": [[5.0, 15.0], [35.0, 45.0], [65.0, 80.0]],  # gapped bins
    "brusselator": [[5.0, 7.5], [5.7, 12.5], [12.5, 17.5], [20.0, 22.5]],  # gapped bins
}

WIDTH = 128
DEPTH = 4
N_TRAIN = 10_000
N_TEST = 2_000
N_T = 200
ITERATIONS = 50_000
LR = 1e-3
LR_DECAY_STEP = 2000  # StepLR: halve lr every this many iterations (None disables decay)
LR_DECAY_GAMMA = 0.5
FOURIER_FREQS = 0  # >0 adds sin/cos encoding of t to the trunk input (see _make_fourier_transform)
FOURIER_MAX_FREQ = 32.0  # highest encoded frequency, in cycles over [0, t_end]
SEED = 0

# Retrain profile: the recipe of the paper's DeepONets (see module docstring).
_RETRAIN_COMMON = {
    "n_train": 8192,
    "n_t": 1000,
    "iterations": 30_000,
    "batch_size": 256,
    "msffn_sigmas": [1.0, 10.0],
}
_WIDE = [1, 256, 256, 256, 256, 256]
RETRAIN_SETTINGS = {
    "scalar_decay": {
        **_RETRAIN_COMMON,
        "branch_layers": [1, 128, 128],
        "trunk_layers": [1, 128, 128, 128],
    },
    "van_der_pol": {**_RETRAIN_COMMON, "branch_layers": _WIDE, "trunk_layers": _WIDE},
    "brusselator": {**_RETRAIN_COMMON, "branch_layers": _WIDE, "trunk_layers": _WIDE},
}


def _load_yaml_env(system: str) -> dict:
    fname = {
        "scalar_decay": "scalar_decay_default.yaml",
        "van_der_pol": "van_der_pol_default.yaml",
        "brusselator": "brusselator_default.yaml",
    }[system]
    path = os.path.join(CONFIGS_ROOT, system, fname)
    with open(path) as f:
        return yaml.safe_load(f)["env"]


def _resolve_bins(system: str, domain: str, cfg: dict):
    test_bins = cfg["test_bins"]
    if domain == "complete":
        return test_bins, test_bins
    if domain == "binned":
        return _BINNED_TRAIN_BINS[system], test_bins
    raise ValueError(domain)


def _select_cached_file(system: str, split: str) -> str:
    """Select the largest cached split that contains full reference paths."""
    data_dir = os.path.join(HERE, "..", "..", "..", "data", system, "evaluation", f"{split}set")
    candidates = []
    for path in glob.glob(os.path.join(data_dir, "*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if (
                "task_params" not in data.files
                or "ref_ts" not in data.files
                or "ref_ys" not in data.files
            ):
                continue
            n = len(data["task_params"])
        candidates.append((n, os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            f"No cached {split}set with task_params/ref_ts/ref_ys found in {data_dir}"
        )
    _, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return selected


def _load_cached_split(system: str, split: str):
    """Load a cached task-parameter/reference-trajectory split."""
    path = _select_cached_file(system, split)
    with np.load(path, allow_pickle=False) as data:
        task_params = np.asarray(data["task_params"], dtype=np.float32).reshape(-1)
        ref_ts = np.asarray(data["ref_ts"], dtype=np.float64)
        ref_ys = np.asarray(data["ref_ys"], dtype=np.float64)

    if ref_ys.ndim == 2:
        ref_ys = ref_ys[:, :, None]
    if ref_ts.ndim != 2 or ref_ys.ndim != 3:
        raise ValueError(
            f"Invalid cached trajectory shapes in {path}: ref_ts={ref_ts.shape}, ref_ys={ref_ys.shape}"
        )
    if ref_ts.shape[0] != len(task_params) or ref_ys.shape[:2] != ref_ts.shape:
        raise ValueError(
            f"Cached task/trajectory size mismatch in {path}: task_params={task_params.shape}, "
            f"ref_ts={ref_ts.shape}, ref_ys={ref_ys.shape}"
        )
    if not np.all(np.isfinite(task_params)):
        raise ValueError(f"Non-finite task parameters in {path}")
    return path, task_params, ref_ts, ref_ys


def _interpolate_cached_trajectories(
    ref_ts: np.ndarray, ref_ys: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    """Interpolate padded, irregular cached paths onto a common time grid."""
    out = np.empty((ref_ts.shape[0], len(t_grid), ref_ys.shape[-1]), dtype=np.float64)
    for i in range(ref_ts.shape[0]):
        finite = np.isfinite(ref_ts[i])
        if not np.any(finite):
            raise ValueError(f"Cached trajectory {i} contains no finite time points")
        t_valid = ref_ts[i, finite]
        y_valid = ref_ys[i, finite]
        for dim in range(ref_ys.shape[-1]):
            out[i, :, dim] = np.interp(t_grid, t_valid, y_valid[:, dim])
    return out


def _build_cached_dataset(system: str, n_t: int, train_bins=None):
    train_path, train_params, train_ts, train_ys = _load_cached_split(system, "train")
    test_path, test_params, test_ts, test_ys = _load_cached_split(system, "test")
    if train_ys.shape[-1] != test_ys.shape[-1]:
        raise ValueError(
            f"Cached train/test state dimensions differ: {train_ys.shape[-1]} vs {test_ys.shape[-1]}"
        )

    if train_bins is not None:
        in_train_bins = np.zeros(len(train_params), dtype=bool)
        for lo, hi in train_bins:
            in_train_bins |= (train_params >= float(lo)) & (train_params <= float(hi))
        if not np.any(in_train_bins):
            raise ValueError(
                f"Cached trainset {train_path} contains no samples in requested bins {train_bins}"
            )
        before = len(train_params)
        train_params = train_params[in_train_bins]
        train_ts = train_ts[in_train_bins]
        train_ys = train_ys[in_train_bins]
        print(
            f"  cache train-bin filter: {before} -> {len(train_params)} samples, bins={train_bins}"
        )

    train_counts = np.isfinite(train_ts).sum(axis=1)
    test_counts = np.isfinite(test_ts).sum(axis=1)
    train_last = train_ts[np.arange(len(train_ts)), train_counts - 1]
    test_last = test_ts[np.arange(len(test_ts)), test_counts - 1]
    cache_t_end = float(max(np.max(train_last), np.max(test_last)))
    if not np.isfinite(cache_t_end) or cache_t_end <= 0:
        raise ValueError(f"Could not determine a valid cached time horizon for {system}")
    if not np.isclose(np.median(train_last), np.median(test_last), rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"Cached train/test horizons differ for {system}: "
            f"train median={np.median(train_last)}, test median={np.median(test_last)}"
        )

    t_grid = np.linspace(0.0, cache_t_end, n_t)
    y_train = _interpolate_cached_trajectories(train_ts, train_ys, t_grid)
    y_test = _interpolate_cached_trajectories(test_ts, test_ys, t_grid)
    branch_train = train_params[:, None]
    branch_test = test_params[:, None]
    num_outputs = train_ys.shape[-1]
    if num_outputs == 1:
        # DeepXDE's single-output target is (N, N_t); cached scalar ref_ys are (N, T, 1).
        y_train = y_train[:, :, 0]
        y_test = y_test[:, :, 0]
    print(f"  cache train: {train_path} (n={len(train_params)})")
    print(f"  cache test:  {test_path} (n={len(test_params)})")
    print(f"  cache horizon: t_end={cache_t_end:g}, interpolated points={n_t}")
    info = {
        "source": "varibad-cache",
        "train_file": os.path.abspath(train_path),
        "test_file": os.path.abspath(test_path),
        "cache_t_end": cache_t_end,
        "train_parameter_min": float(np.min(train_params)),
        "train_parameter_max": float(np.max(train_params)),
        "test_parameter_min": float(np.min(test_params)),
        "test_parameter_max": float(np.max(test_params)),
    }
    return (
        num_outputs,
        branch_train.astype(np.float32),
        y_train.astype(np.float32),
        branch_test.astype(np.float32),
        y_test.astype(np.float32),
        t_grid,
        info,
    )


def _build_dataset(
    system: str,
    cfg: dict,
    train_bins,
    test_bins,
    n_train: int,
    n_test: int,
    n_t: int,
    seed: int,
    data_device: str = "cpu",
    dataset_source: str = "generated",
):
    """Return arrays ready for ``dde.data.TripleCartesianProd`` plus metadata."""
    if dataset_source == "cache":
        return _build_cached_dataset(system, n_t, train_bins=train_bins)
    if dataset_source != "generated":
        raise ValueError(f"Unknown dataset source: {dataset_source}")

    rng = np.random.default_rng(seed)
    t_end = float(cfg["t_end"])
    t_grid = np.linspace(0.0, t_end, n_t)

    if system == "scalar_decay":
        scheme = cfg.get("task_sample_scheme", "log-binned")
        lam_train = sample_binned_values(rng, train_bins, n_train, scheme=scheme)
        lam_test = sample_binned_values(rng, test_bins, n_test, scheme=scheme)
        y_train = scalar_decay_trajectories_on_device(lam_train, t_grid, device=data_device)
        y_test = scalar_decay_trajectories_on_device(lam_test, t_grid, device=data_device)
        branch_train = lam_train[:, None].astype(np.float32)
        branch_test = lam_test[:, None].astype(np.float32)
        num_outputs = NUM_OUTPUTS[system]
    elif system in ("van_der_pol", "brusselator"):
        tasks_train = sample_tasks(system, rng, cfg, train_bins, n_train)
        tasks_test = sample_tasks(system, rng, cfg, test_bins, n_test)
        y_train = generate_trajectories(system, tasks_train, t_grid, t_end, device=data_device)
        y_test = generate_trajectories(system, tasks_test, t_grid, t_end, device=data_device)
        branch_train = tasks_train[:, :1].astype(np.float32)
        branch_test = tasks_test[:, :1].astype(np.float32)
        num_outputs = NUM_OUTPUTS[system]
    else:
        raise ValueError(system)

    return (
        num_outputs,
        branch_train,
        y_train.astype(np.float32),
        branch_test,
        y_test.astype(np.float32),
        t_grid,
        {"source": "generated", "data_device": data_device},
    )


def _make_fourier_transform(n_freqs: int, max_freq: float, t_end: float):
    """Sin/cos encoding of the time input at log-spaced frequencies up to max_freq
    cycles over [0, t_end] (Tancik et al. 2020), to counter the spectral bias of MLPs.
    """
    import torch

    octaves = torch.linspace(0.0, float(np.log2(max(max_freq, 1.0))), n_freqs)
    freqs = (2.0**octaves) / t_end

    def transform(t):
        args = 2.0 * np.pi * t * freqs.to(t.device)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return transform


def _build_msffn_trunk(layer_sizes, activation, sigmas):
    """Multi-scale Fourier feature trunk (MsFFN; Wang, Wang & Perdikaris 2021).

    PyTorch port of DeepXDE's paddle/tensorflow_compat_v1 MsFFN. Each scale sigma
    projects the input through a fixed Gaussian matrix ~ N(0, sigma) into
    [cos, sin] features, which pass through shared hidden layers; the outputs of
    all scales are concatenated and mapped to out_dim. Assigned to net.trunk after
    construction, since DeepXDE's PyTorch backend has no MsFFN.
    """
    import torch

    in_dim = layer_sizes[0]
    fourier_dim = layer_sizes[1]
    if fourier_dim % 2 != 0:
        raise ValueError(f"MsFFN fourier embedding width must be even, got {fourier_dim}")
    hidden_sizes = layer_sizes[2:-1]
    out_dim = layer_sizes[-1]
    sigmas = list(sigmas)

    class _MsFFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = activation
            for i, sigma in enumerate(sigmas):
                self.register_buffer(f"b{i}", torch.randn(in_dim, fourier_dim // 2) * sigma)
            self.hidden = torch.nn.ModuleList()
            prev = fourier_dim
            for size in hidden_sizes:
                self.hidden.append(torch.nn.Linear(prev, size))
                prev = size
            self.out = torch.nn.Linear(prev * len(sigmas), out_dim)

        def _fourier(self, x, b):
            proj = x @ b
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        def _shared_fc(self, y):
            for linear in self.hidden:
                y = self.act(linear(y))
            return y

        def forward(self, x):
            feats = [
                self._shared_fc(self._fourier(x, getattr(self, f"b{i}")))
                for i in range(len(sigmas))
            ]
            return self.out(torch.cat(feats, dim=-1))

    return _MsFFN()


def build_deeponet(
    branch_layers,
    trunk_layers,
    num_outputs,
    *,
    msffn_sigmas=None,
    fourier_freqs=0,
    fourier_max_freq=FOURIER_MAX_FREQ,
    t_end=None,
    verbose=True,
):
    """DeepONetCartesianProd with an optional sin/cos Fourier or multi-scale Fourier (MsFFN) trunk.

    Returns (net, trunk_layers), since fourier_freqs > 0 resizes trunk_layers[0].
    """
    import deepxde as dde
    import torch

    trunk_layers = list(trunk_layers)
    if msffn_sigmas:
        if fourier_freqs > 0:
            print("  msffn_sigmas given; ignoring fourier_freqs (mutually exclusive trunk options)")
        fourier_freqs = 0
    elif fourier_freqs > 0:
        trunk_layers = [2 * fourier_freqs] + list(trunk_layers[1:])
    net_kwargs = (
        dict(num_outputs=num_outputs, multi_output_strategy="independent")
        if num_outputs > 1
        else {}
    )
    net = dde.nn.DeepONetCartesianProd(
        list(branch_layers), trunk_layers, "relu", "Glorot normal", **net_kwargs
    )
    if fourier_freqs > 0:
        if t_end is None:
            raise ValueError("t_end is required to rebuild the raw sin/cos Fourier trunk transform")
        net.apply_feature_transform(_make_fourier_transform(fourier_freqs, fourier_max_freq, t_end))
        if verbose:
            print(
                f"  fourier features: {fourier_freqs} freqs, max_freq={fourier_max_freq} cycles/t_end "
                f"(trunk input dim {trunk_layers[0]})"
            )
    if msffn_sigmas:
        activation_fn = dde.nn.activations.get("relu")
        # Multi-output nets hold one trunk per output (net.trunk[i]).
        if num_outputs > 1:
            net.trunk = torch.nn.ModuleList(
                _build_msffn_trunk(trunk_layers, activation_fn, msffn_sigmas)
                for _ in range(num_outputs)
            )
        else:
            net.trunk = _build_msffn_trunk(trunk_layers, activation_fn, msffn_sigmas)
        if verbose:
            print(
                f"  msffn trunk: sigmas={msffn_sigmas} (fourier width={trunk_layers[1]}, "
                f"hidden={trunk_layers[2:-1]}, out={trunk_layers[-1]}, num_outputs={num_outputs})"
            )
    return net, trunk_layers


def train_one(
    system: str,
    domain: str,
    *,
    width=WIDTH,
    depth=DEPTH,
    n_train=N_TRAIN,
    n_test=N_TEST,
    n_t=N_T,
    iterations=None,
    epochs=None,
    batch_size=None,
    lr=LR,
    seed=SEED,
    verbose_every=1000,
    profile="unified",
    output_tag=None,
    data_device="cpu",
    branch_layers=None,
    trunk_layers=None,
    dataset_source="generated",
    lr_decay_step=LR_DECAY_STEP,
    lr_decay_gamma=LR_DECAY_GAMMA,
    fourier_freqs=FOURIER_FREQS,
    fourier_max_freq=FOURIER_MAX_FREQ,
    msffn_sigmas=None,
):
    import deepxde as dde

    cfg = _load_yaml_env(system)
    train_bins, test_bins = _resolve_bins(system, domain, cfg)
    print(
        f"[*] {system}/{domain}: train_bins={train_bins}  test_bins={test_bins}  t_end={cfg['t_end']}"
    )

    t0 = time.perf_counter()
    num_outputs, branch_train, y_train, branch_test, y_test, t_grid, dataset_info = _build_dataset(
        system,
        cfg,
        train_bins,
        test_bins,
        n_train,
        n_test,
        n_t,
        seed,
        data_device=data_device,
        dataset_source=dataset_source,
    )
    actual_n_train = len(branch_train)
    actual_n_test = len(branch_test)
    if dataset_source == "cache" and (actual_n_train != n_train or actual_n_test != n_test):
        print(
            f"  cache sizes override requested sizes: requested train/test={n_train}/{n_test}, "
            f"loaded train/test={actual_n_train}/{actual_n_test}"
        )
    print(
        f"  data gegenereerd in {time.perf_counter() - t0:.1f}s "
        f"(n_train={actual_n_train}, n_test={actual_n_test}, source={dataset_source}, device={data_device})"
    )

    t_grid_col = t_grid[:, None].astype(np.float32)
    data = dde.data.TripleCartesianProd(
        X_train=(branch_train, t_grid_col),
        y_train=y_train,
        X_test=(branch_test, t_grid_col),
        y_test=y_test,
    )
    if branch_layers is None:
        branch_layers = [1] + [width] * depth
    if trunk_layers is None:
        trunk_layers = [1] + [width] * depth
    net, trunk_layers = build_deeponet(
        branch_layers,
        trunk_layers,
        num_outputs,
        msffn_sigmas=msffn_sigmas,
        fourier_freqs=fourier_freqs,
        fourier_max_freq=fourier_max_freq,
        t_end=cfg["t_end"],
    )
    model = dde.Model(data, net)
    decay = ("step", lr_decay_step, lr_decay_gamma) if lr_decay_step else None
    model.compile("adam", lr=lr, decay=decay)
    if decay is not None:
        print(f"  lr decay: step={lr_decay_step} gamma={lr_decay_gamma} (initial lr={lr})")

    epochs_drove_iterations = False
    if iterations is not None:
        # An explicit iteration count takes precedence over epochs.
        display_every = max(1, min(verbose_every, iterations))
    elif epochs is not None:
        if batch_size is None:
            raise ValueError("batch_size must be provided when epochs is used")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        iters_per_epoch = (actual_n_train + batch_size - 1) // batch_size
        iterations = epochs * iters_per_epoch
        display_every = iters_per_epoch
        epochs_drove_iterations = True
    else:
        iterations = ITERATIONS
        display_every = max(1, min(verbose_every, iterations))

    out_dir = os.path.join(HERE, "output-deeponet", system)
    models_dir = os.path.join(out_dir, "models")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    tag = output_tag or f"{system}_{domain}"
    ckpt_dir = os.path.join(models_dir, f"deeponet_{tag}_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    class _BestTestLossCheckpoint(dde.callbacks.ModelCheckpoint):
        """ModelCheckpoint that monitors test loss and remembers the saved path, so
        train_one can restore the best weights before the final save.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_path = None

        def on_epoch_end(self):
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save < self.period:
                return
            self.epochs_since_last_save = 0
            if not self.save_better_only:
                self.model.save(self.filepath, verbose=self.verbose)
                return
            current = self.get_monitor_value()
            if self.monitor_op(current, self.best):
                self.best_path = self.model.save(self.filepath, verbose=0)
                self.best = current

    checkpointer = _BestTestLossCheckpoint(
        os.path.join(ckpt_dir, "model"),
        save_better_only=True,
        period=max(1, iterations // 20),
        monitor="test loss",
    )

    if epochs_drove_iterations:
        print(
            f"  epochs: {epochs} x {iters_per_epoch} iterations/epoch = {iterations} iterations, "
            f"batch_size={batch_size}"
        )
    elif iterations is not None and epochs is not None:
        print(
            f"  iterations override: {iterations} iterations (ignoring epochs={epochs}), "
            f"batch_size={batch_size}"
        )
    print(
        f"  net: branch={branch_layers}, trunk={trunk_layers}, num_outputs={num_outputs}, "
        f"iterations={iterations}"
    )
    t0 = time.perf_counter()
    train_kwargs = dict(
        iterations=iterations,
        callbacks=[checkpointer],
        display_every=display_every,
    )
    if batch_size is not None:
        train_kwargs["batch_size"] = batch_size
    losshistory, _ = model.train(**train_kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  training time: {elapsed / 60:.1f} min")

    final_test_loss = sum(losshistory.loss_test[-1])
    if checkpointer.best_path is not None and checkpointer.best < final_test_loss:
        print(
            f"  restoring best test-loss checkpoint: {checkpointer.best_path} "
            f"(test loss {checkpointer.best:.4e} < final-iteration test loss {final_test_loss:.4e})"
        )
        model.restore(checkpointer.best_path, verbose=1)
    else:
        print(
            f"  final iteration is already the best test-loss checkpoint "
            f"(test loss {final_test_loss:.4e})"
        )

    # restore() does not reset the iteration counter, so the final checkpoint keeps
    # the '..._final-{iterations}' name even when restoring an earlier step.
    final_path = model.save(os.path.join(models_dir, f"deeponet_{tag}_final"))
    print(f"  saved model: {final_path}")

    y_pred = model.predict((branch_test, t_grid_col))
    abs_err = np.abs(y_pred - y_test)
    print(f"  test mean|error|={abs_err.mean():.4e}  max|error|={abs_err.max():.4e}")

    np.savez(
        os.path.join(results_dir, f"deeponet_{tag}_losshistory.npz"),
        steps=np.array(losshistory.steps),
        loss_train=np.array(losshistory.loss_train),
        loss_test=np.array(losshistory.loss_test),
        train_bins=np.asarray(train_bins, dtype=float),
        test_bins=np.asarray(test_bins, dtype=float),
        mean_abs_err=abs_err.mean(),
        max_abs_err=abs_err.max(),
        n_train=actual_n_train,
        n_test=actual_n_test,
        n_t=n_t,
        epochs=-1 if epochs is None else epochs,
        batch_size=-1 if batch_size is None else batch_size,
        iterations=iterations,
    )
    with open(
        os.path.join(results_dir, f"deeponet_{tag}_training_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "system": system,
                "domain": domain,
                "profile": profile,
                "dataset_source": dataset_source,
                "seed": seed,
                "n_train": actual_n_train,
                "n_test": actual_n_test,
                "n_t": n_t,
                "epochs": epochs,
                "batch_size": batch_size,
                "iterations": iterations,
                "learning_rate": lr,
                "lr_decay_step": lr_decay_step,
                "lr_decay_gamma": lr_decay_gamma,
                "fourier_freqs": fourier_freqs,
                "fourier_max_freq": fourier_max_freq if fourier_freqs > 0 else None,
                "msffn_sigmas": msffn_sigmas,
                "data_device": data_device,
                "branch_layers": branch_layers,
                "trunk_layers": trunk_layers,
                "train_bins": train_bins,
                "test_bins": test_bins,
                "dataset_info": dataset_info,
            },
            f,
            indent=2,
        )
    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--all", action="store_true", help="train all 3 systems x 2 domains")
    parser.add_argument(
        "--profile",
        choices=["unified", "retrain"],
        default="unified",
        help="unified=one shared WIDTH x DEPTH architecture; retrain=the paper recipe (RETRAIN_SETTINGS)",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--dataset-source",
        choices=["generated", "cache"],
        default="generated",
        help="generated=sample/integrate trajectories; cache=load VariBAD train/test trajectories",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="override iterations in the unified profile; in the retrain "
        "profile, hard-caps total iterations and takes precedence over --epochs",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="override training-set size; retrain uses this for both train and test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="retrain profile: train for this many epochs instead of its fixed iterations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="override batch size in the retrain profile"
    )
    parser.add_argument(
        "--lr-decay-step",
        type=int,
        default=LR_DECAY_STEP,
        help="halve (x gamma) the lr every N iterations; 0 disables decay",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=LR_DECAY_GAMMA,
        help="multiplicative factor applied to lr every --lr-decay-step iterations",
    )
    parser.add_argument(
        "--fourier-freqs",
        type=int,
        default=FOURIER_FREQS,
        help="sin/cos positional-encode the trunk's t input with this many "
        "log-spaced frequencies (0 disables, uses raw t)",
    )
    parser.add_argument(
        "--fourier-max-freq",
        type=float,
        default=FOURIER_MAX_FREQ,
        help="highest encoded frequency, in cycles over the full [0, t_end] domain",
    )
    parser.add_argument(
        "--msffn-sigmas",
        type=str,
        default=None,
        help="comma-separated list of Gaussian projection std devs for a "
        "multi-scale Fourier feature trunk (Wang/Wang/Perdikaris 2021), "
        "e.g. '1,10'; overrides --fourier-freqs when set",
    )
    args = parser.parse_args()
    msffn_sigmas = [float(s) for s in args.msffn_sigmas.split(",")] if args.msffn_sigmas else None

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("DDE_BACKEND", "pytorch")

    if args.profile == "retrain":
        if args.all:
            systems = SYSTEMS
        elif args.system:
            systems = (args.system,)
        else:
            parser.error("retrain profile requires --system or --all")
        domains = (args.domain or "complete",)
    else:
        if args.all:
            combos = [(s, d) for s in SYSTEMS for d in DOMAINS]
        elif args.system and args.domain:
            combos = [(args.system, args.domain)]
        else:
            parser.error("unified profile requires --system + --domain, or --all")

    if args.profile == "retrain":
        combos = [(s, d) for s in systems for d in domains]

    for system, domain in combos:
        print(f"\n{'=' * 70}\n[{system}/{domain}]\n{'=' * 70}")
        if args.profile == "retrain":
            spec = RETRAIN_SETTINGS[system]
            n_train = args.n_train or spec["n_train"]
            train_one(
                system,
                domain,
                n_train=n_train,
                n_test=n_train,
                n_t=spec["n_t"],
                iterations=args.iterations or (None if args.epochs else spec["iterations"]),
                epochs=None if args.iterations else args.epochs,
                batch_size=args.batch_size or spec["batch_size"],
                seed=SEED,
                profile="retrain",
                output_tag=f"{system}_retrain_{domain}",
                data_device="cuda" if args.device == "gpu" else "cpu",
                branch_layers=spec["branch_layers"],
                trunk_layers=spec["trunk_layers"],
                dataset_source=args.dataset_source,
                lr_decay_step=args.lr_decay_step or None,
                lr_decay_gamma=args.lr_decay_gamma,
                fourier_freqs=args.fourier_freqs,
                fourier_max_freq=args.fourier_max_freq,
                msffn_sigmas=msffn_sigmas or spec["msffn_sigmas"],
            )
        else:
            train_one(
                system,
                domain,
                iterations=args.iterations or ITERATIONS,
                n_train=args.n_train or N_TRAIN,
                data_device="cuda" if args.device == "gpu" else "cpu",
                dataset_source=args.dataset_source,
                lr_decay_step=args.lr_decay_step or None,
                lr_decay_gamma=args.lr_decay_gamma,
                fourier_freqs=args.fourier_freqs,
                fourier_max_freq=args.fourier_max_freq,
                msffn_sigmas=msffn_sigmas,
            )


if __name__ == "__main__":
    main()
