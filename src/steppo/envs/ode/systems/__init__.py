"""Registry of ODE systems for the generic ODEEnv.

Each system provides the mathematical problem (RHS, initial conditions,
observation) while ODEEnv owns the solver mechanics (Kvaerno5, accept/reject,
step-size action).

Adding a new system:
  1. Create src/steppo/envs/ode/systems/<name>.py with a module-level SPEC.
  2. Register it in ODE_SYSTEMS below.
"""

from dataclasses import dataclass
from typing import Callable

from jax import Array
from jax.random import PRNGKey


@dataclass
class ODESystemSpec:
    """All ODE-system-specific logic, expressed as pure functions.

    Each callable is JAX-traceable (no Python control flow depending on
    traced values) so they compose cleanly with vmap and lax.scan.

    Args to rhs/sample_task/y0/obs are concrete at trace time except where noted.
    """

    name: str
    y_dim: int  # dimensionality of the ODE state vector
    obs_dim: int  # default obs dimensionality (all features enabled)

    # f(t, y, task) → ẏ   — task is a 1-D Array of ODE parameters
    rhs: Callable[[Array, Array, Array], Array]

    # (key, config) → task Array of shape (param_dim,)
    sample_task: Callable[[PRNGKey, object], Array]

    # (task, key, config) → y0 Array of shape (y_dim,)
    y0: Callable[[Array, PRNGKey, object], Array]

    # (state, config, max_steps) → obs Array of shape (obs_dim,)
    obs: Callable[[object, object, int], Array]

    # feature_name → number of observation dimensions (for dynamic obs_dim)
    feature_dims: dict = None

    def get_obs_dim(self, config=None) -> int:
        """Return the observation width for the selected feature subset."""
        if config is None or self.feature_dims is None:
            return self.obs_dim
        unknown = [f for f in config.obs_features if f not in self.feature_dims]
        if unknown:
            raise ValueError(
                f"obs_features contains unrecognized feature(s) {unknown} for "
                f"system '{self.name}' — known features: {sorted(self.feature_dims)}."
            )
        return sum(self.feature_dims[f] for f in config.obs_features)


# ── Registry ──────────────────────────────────────────────────────────────────


def _load_systems() -> dict[str, ODESystemSpec]:
    """Load the built-in ODE system specifications."""
    from steppo.envs.ode.systems.brusselator import SPEC as _brus
    from steppo.envs.ode.systems.chemical_cascade import SPEC as _chem
    from steppo.envs.ode.systems.chua import SPEC as _chua
    from steppo.envs.ode.systems.chua import SPEC_SMOOTH as _chua_smooth
    from steppo.envs.ode.systems.fitzhugh_nagumo import SPEC as _fhn
    from steppo.envs.ode.systems.fosm import SPEC as _fosm
    from steppo.envs.ode.systems.fosm import SPEC_SMOOTH as _fosm_smooth
    from steppo.envs.ode.systems.robertson import SPEC as _rob
    from steppo.envs.ode.systems.scalar_decay import SPEC as _scalar
    from steppo.envs.ode.systems.van_der_pol import SPEC as _vdp

    return {
        "scalar_decay": _scalar,
        "van_der_pol": _vdp,
        "robertson": _rob,
        "brusselator": _brus,
        "fitzhugh_nagumo": _fhn,
        "chemical_cascade": _chem,
        "fosm": _fosm,
        "fosm_smooth": _fosm_smooth,
        "chua": _chua,
        "chua_smooth": _chua_smooth,
    }


def get_system(name: str) -> ODESystemSpec:
    """Return the registered ODE system named by ``name``."""
    systems = _load_systems()
    if name not in systems:
        raise ValueError(f"Unknown ODE system '{name}'. Available: {sorted(systems)}")
    return systems[name]


def get_rhs(config) -> Callable[[Array, Array, Array], Array]:
    """Return the configured RHS, including any optional pulse forcing."""
    spec = get_system(config.system)
    if config.system == "chemical_cascade":
        from steppo.envs.ode.systems.chemical_cascade import make_rhs

        return make_rhs(config)
    if config.system == "fitzhugh_nagumo":
        from steppo.envs.ode.systems.fitzhugh_nagumo import make_rhs

        return make_rhs(config)
    if config.system == "scalar_decay":
        from steppo.envs.ode.systems.scalar_decay import make_rhs

        return make_rhs(config)
    if config.system == "fosm_smooth":
        from steppo.envs.ode.systems.fosm import make_rhs

        return make_rhs(config)
    if config.system == "chua_smooth":
        from steppo.envs.ode.systems.chua import make_rhs

        return make_rhs(config)
    return spec.rhs
