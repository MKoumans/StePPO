"""Per-system task parameter ξ: its config bound fields and display symbol."""

# system -> (lo field, hi field, symbol)
_TASK_PARAMS = {
    "scalar_decay": ("lam_min", "lam_max", "λ"),
    "van_der_pol": ("mu_min", "mu_max", "μ"),
    "robertson": ("k2_min", "k2_max", "k2"),
    "brusselator": ("B_min", "B_max", "B"),
    "fitzhugh_nagumo": ("eps_min", "eps_max", "ε"),
    "chemical_cascade": ("cc_lam_min", "cc_lam_max", "λ"),
    "fosm": ("D_min", "D_max", "D"),
    "fosm_smooth": ("D_min", "D_max", "D"),
    "chua": ("m_min", "m_max", "m"),
    "chua_smooth": ("m_min", "m_max", "m"),
}


def task_bounds(env_config) -> tuple[float, float]:
    """Raw (lo, hi) task-parameter bounds of `env_config.system`."""
    lo_attr, hi_attr, _ = _TASK_PARAMS.get(env_config.system, ("lam_min", "lam_max", None))
    return float(getattr(env_config, lo_attr)), float(getattr(env_config, hi_attr))


def task_label(system: str) -> str:
    """Display symbol of `system`'s task parameter (e.g. "μ"), or "param" if unregistered."""
    fields = _TASK_PARAMS.get(system)
    return fields[2] if fields is not None else "param"
