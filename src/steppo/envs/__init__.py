"""Environment factories and implementations."""


def make_env(name: str, config=None):
    """Create an environment and its default parameters for ``name``."""
    if name in ("ode", "scalar_decay", "van_der_pol"):
        from steppo.configs.base_config import ODEEnvConfig
        from steppo.envs.ode import ODEEnv, ODEParams

        env_config = config.env if config is not None else ODEEnvConfig(system="scalar_decay")
        # Allow name to override system when called without a config
        if config is None and name != "ode":
            env_config = ODEEnvConfig(system=name)
        if config is not None and config.episode_max_steps > 0:
            max_steps = config.episode_max_steps
        elif config is not None:
            max_steps = config.rollout_steps
        else:
            max_steps = 200
        env = ODEEnv(env_config, max_steps=max_steps)
        params = ODEParams(lam=1.0)
        return env, params
    raise ValueError(f"Unknown env: {name}. Available: ode, scalar_decay, van_der_pol")
