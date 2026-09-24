"""Upload a DeepONet checkpoint to MKoumans/deeponet in the layout paper/utils.py reads.

Each (system, domain) run is one branch, e.g. vdp-complete, holding the raw DeepXDE
checkpoint as its only *.pt file plus a metadata.json.

Run in the DeepONet container from the repo root (HF_TOKEN in the environment):
    PYTHONPATH=. python research/baselines/deeponet/deeponet_hub.py \\
        --checkpoint research/baselines/deeponet/output-deeponet/van_der_pol/models/deeponet_van_der_pol_retrain_complete_final-30000.pt \\
        --config research/baselines/deeponet/output-deeponet/van_der_pol/results/deeponet_van_der_pol_retrain_complete_training_config.json \\
        --staging
"""

import argparse
import json
from pathlib import Path

from train_deeponet_unified import NUM_OUTPUTS, build_deeponet

REPO_ID = "MKoumans/deeponet"
STAGING_SUFFIX = "-staging"
_SYSTEM_PREFIX = {"scalar_decay": "sd", "van_der_pol": "vdp", "brusselator": "bru"}


def default_revision(config: dict) -> str:
    return f"{_SYSTEM_PREFIX[config['system']]}-{config['domain']}"


def build_metadata(config: dict) -> dict:
    keys = (
        "system",
        "domain",
        "branch_layers",
        "trunk_layers",
        "msffn_sigmas",
        "n_train",
        "n_test",
        "iterations",
        "train_bins",
        "test_bins",
    )
    return {"num_outputs": NUM_OUTPUTS[config["system"]], **{k: config[k] for k in keys}}


def validate_checkpoint(checkpoint: Path, config: dict) -> None:
    """Raise unless the checkpoint loads strictly into the net described by `config`."""
    import torch

    net, _ = build_deeponet(
        config["branch_layers"],
        config["trunk_layers"],
        NUM_OUTPUTS[config["system"]],
        msffn_sigmas=config.get("msffn_sigmas"),
        fourier_freqs=config.get("fourier_freqs") or 0,
        fourier_max_freq=config.get("fourier_max_freq"),
        t_end=config.get("dataset_info", {}).get("cache_t_end"),
        verbose=False,
    )
    net.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])


def upload(
    checkpoint: Path, config: dict, repo_id: str, revision: str, commit_message: str | None = None
) -> list[str]:
    """Commit the checkpoint and its metadata.json to `revision`, replacing any other *.pt there."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    validate_checkpoint(checkpoint, config)

    api = HfApi()
    api.create_branch(repo_id, branch=revision, exist_ok=True)
    # paper/utils.py unpacks exactly one *.pt per branch.
    stale = [
        f
        for f in api.list_repo_files(repo_id, revision=revision)
        if f.endswith(".pt") and f != checkpoint.name
    ]
    operations = [
        CommitOperationAdd(path_in_repo=checkpoint.name, path_or_fileobj=str(checkpoint)),
        CommitOperationAdd(
            path_in_repo="metadata.json",
            path_or_fileobj=(json.dumps(build_metadata(config), indent=2) + "\n").encode(),
        ),
        *(CommitOperationDelete(path_in_repo=f) for f in stale),
    ]
    api.create_commit(
        repo_id,
        operations=operations,
        revision=revision,
        commit_message=commit_message or f"Upload {checkpoint.name}",
    )
    return stale


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="deeponet_<tag>_final-<iterations>.pt"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the matching deeponet_<tag>_training_config.json",
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", help="branch to commit to (default: sd-/vdp-/bru-<domain>)")
    parser.add_argument(
        "--staging", action="store_true", help=f"append {STAGING_SUFFIX!r} to the branch name"
    )
    parser.add_argument("--commit-message")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    revision = args.revision or default_revision(config)
    if args.staging:
        revision += STAGING_SUFFIX
    stale = upload(args.checkpoint, config, args.repo_id, revision, args.commit_message)
    print(
        f"[+] {args.repo_id}@{revision}: {args.checkpoint.name}"
        + (f" (removed {stale})" if stale else "")
    )


if __name__ == "__main__":
    main()
