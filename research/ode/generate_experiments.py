"""Generate a factorial sweep of experiment configs from a base config + sweep spec.

The sweep spec is a small YAML file mapping dotted TrainConfig paths to the
values to sweep over, e.g.:

    model: vae_vdp_1, vae_vdp_2
    warmstart.enabled: true, false
    vae.sequential_kl: true, false
    model.latent_dim: 1, 2, 4, 8, 16, 32

A bare `model` key swaps the whole referenced architecture file (configs/models/<name>.yaml).
A `model.<field>` key instead overrides one field that architecture file would otherwise
supply as a default (latent_dim, encoder.*, decoder.*, policy.*) — it is rewritten to the
env-config path that field is merged into (vae.latent_dim, vae.encoder.*, vae.decoder.*,
ppo.policy.*) via `_merge_model_ref` in src/steppo/configs/base_config.py, since an explicit
env-config value always takes precedence over the model file's default.

Each key's values can be given as a comma-separated string or a YAML list.

A key may also be a comma-separated list of dotted paths (e.g.
"ppo.entropy_coeff_start,ppo.entropy_coeff") to sweep several fields together
as one axis instead of factorial-combining them. Its values are then a YAML
list of same-length lists/tuples, one per combo, e.g.:

    ppo.entropy_coeff_start,ppo.entropy_coeff:
      - [0.0, 0.005]
      - [0.05, 0.01]
      - [0.15, 0.02]

The generator takes the cartesian product of all keys (linked keys count as
a single axis) and writes one config file per combination, named
<prefix>_e<N>.yaml, alongside the base config.
Every generated file is the base config with ONLY the swept keys overridden
(plus exp_name) — all other tuning lives in the base config.

Usage:
    python research/ode/generate_experiments.py \\
        --base configs/envs/ode/van_der_pol/van_der_pol_default.yaml \\
        --sweep configs/envs/ode/van_der_pol/van_der_pol_experiment.yml \\
        --prefix van_der_pol
"""

import argparse
import itertools
import os

import yaml


def parse_args():
    """Parse the base config, sweep specification, and output options."""
    parser = argparse.ArgumentParser(description="Generate factorial experiment configs")
    parser.add_argument("--base", required=True, help="Path to base config YAML")
    parser.add_argument("--sweep", required=True, help="Path to sweep spec YAML")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. van_der_pol")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for generated configs (default: same dir as --base)",
    )
    return parser.parse_args()


def parse_sweep_values(raw):
    """A sweep value is either a YAML list already, or a comma-separated string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [yaml.safe_load(tok.strip()) for tok in raw.split(",")]
    # Single scalar (int/float/bool) sweeping over just one value.
    return [raw]


def set_dotted(data: dict, dotted_key: str, value) -> None:
    """Set data[a][b][c] = value for dotted_key 'a.b.c', creating dicts as needed."""
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


MODEL_FIELD_TARGETS = {
    "latent_dim": "vae.latent_dim",
    "encoder": "vae.encoder",
    "decoder": "vae.decoder",
    "policy": "ppo.policy",
}


def resolve_sweep_key(dotted_key: str) -> str:
    """Rewrite a `model.<field>...` sweep key to its post-merge env-config path.

    A bare `model` key (swapping the whole arch file) passes through unchanged.
    """
    if dotted_key == "model" or not dotted_key.startswith("model."):
        return dotted_key
    field, _, rest = dotted_key[len("model.") :].partition(".")
    if field not in MODEL_FIELD_TARGETS:
        raise ValueError(
            f"Unknown model field '{field}' in sweep key '{dotted_key}'. "
            f"Known fields: {sorted(MODEL_FIELD_TARGETS)}"
        )
    target = MODEL_FIELD_TARGETS[field]
    return f"{target}.{rest}" if rest else target


def main():
    """Generate one config file for each point in the requested sweep."""
    args = parse_args()

    with open(args.base, "r") as f:
        base_data = yaml.safe_load(f)

    with open(args.sweep, "r") as f:
        sweep_spec = yaml.safe_load(f)

    keys = list(sweep_spec.keys())
    value_lists = [parse_sweep_values(sweep_spec[k]) for k in keys]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.base))
    os.makedirs(out_dir, exist_ok=True)

    combos = list(itertools.product(*value_lists))
    written = []
    for i, combo in enumerate(combos, start=1):
        eid = f"e{i}"
        data = yaml.safe_load(yaml.safe_dump(base_data))  # deep copy

        header_lines = [f"# Experiment {eid}: generated from {os.path.basename(args.base)}"]
        for key, value in zip(keys, combo):
            if "," in key:
                subkeys = [k.strip() for k in key.split(",")]
                if not isinstance(value, (list, tuple)) or len(value) != len(subkeys):
                    raise ValueError(
                        f"Linked sweep key '{key}' expects each value to be a list of "
                        f"{len(subkeys)} elements, got {value!r}"
                    )
                for subkey, subvalue in zip(subkeys, value):
                    resolved_subkey = resolve_sweep_key(subkey)
                    set_dotted(data, resolved_subkey, subvalue)
                    if resolved_subkey != subkey:
                        header_lines.append(f"#   {subkey} = {subvalue}  (-> {resolved_subkey})")
                    else:
                        header_lines.append(f"#   {subkey} = {subvalue}")
            else:
                resolved_key = resolve_sweep_key(key)
                set_dotted(data, resolved_key, value)
                if resolved_key != key:
                    header_lines.append(f"#   {key} = {value}  (-> {resolved_key})")
                else:
                    header_lines.append(f"#   {key} = {value}")

        data["exp_name"] = f"{args.prefix}_{eid}"

        out_path = os.path.join(out_dir, f"{args.prefix}_{eid}.yaml")
        with open(out_path, "w") as f:
            f.write("\n".join(header_lines) + "\n")
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

        written.append(out_path)
        print(f"[+] Wrote {out_path}")

    print(
        f"\nGenerated {len(written)} configs ({' x '.join(str(len(v)) for v in value_lists)} factorial)."
    )


if __name__ == "__main__":
    main()
