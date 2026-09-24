# Architecture

## Formulation

Step-size selection is cast as a partially observable MDP. Each ODE instance —
a draw of the hidden task parameter (λ, μ, B, ε, k₂, …) — is an unknown task.
Following VariBAD ([Zintgraf et al., ICLR 2020](https://arxiv.org/abs/1910.08348)),
the agent maintains a belief posterior `q(z | τ)` over that task from the stream
of solver observations, and conditions a PPO actor-critic on the belief rather
than on a point estimate.

```
Trajectory τ = (a₀, s₁, r₀, ..., sₜ)
         │
    ┌────▼──────────┐
    │    Encoder    │ → μ_z, log σ²_z   (posterior q(z | τ))
    └────┬──────────┘
         │ reparameterization
         z ~ N(μ_z, σ²_z)
         │
    ┌────▼──────────┐
    │   Decoders    │ → r̂, ŝ', m̂       (reward, state, task reconstruction)
    └───────────────┘

    [sₜ, zₜ] → ActorCritic π(a | s, z) → aₜ
```

## Two pluggable axes

`Trainer` composes two independent axes, deliberately handled differently:

| Axis | Selected by | Mechanism | Registry |
| --- | --- | --- | --- |
| **Belief model** | `backbone:` | `Trainer` subclass | `src/steppo/models/backbones.py` |
| **Policy algorithm** | `algo:` | injected class of staticmethods | `src/steppo/training/policy_algos.py` |

The belief model varies rarely and is tied to trainer identity. The policy
algorithm must combine freely with any belief model, so subclassing both
axes would multiply combinatorially.

### Registered belief-model backbones

| `backbone:` | Model | Trainer | Config section | Notes |
| --- | --- | --- | --- | --- |
| `varibad` (default) | `VariBADVAE` (`models/vae.py`) | `VariBADTrainer` | `vae:` | GRU (or per-step MLP) over embedded `(action, state, reward[, task])`, full ELBO |

### Registered policy algorithms

| `algo:` | Class | Config section |
| --- | --- | --- |
| `ppo` (default, only) | `PPOAlgorithm` (`training/ppo_algorithm.py`) | `ppo:` |

## Encoder

`models/encoder.py` (varibad backbone). `encoder.encoder_type` selects the
recurrent cell:

- `gru` (default) — GRU with a single hidden state of width `hidden_size`.
- `linear` — a per-step MLP (`linear_hidden_dims`) instead of a recurrent cell,
  as a memory ablation.

`encoder.encoder_inputs` selects which per-step signals are embedded, any subset
of `("action", "state", "reward", "task")`. The base dataclass default is the
original VariBAD triple; `TrainConfig` overrides it to also include `"task"` (the
true hidden parameter), since every registered ODE env exposes one.

## Decoders

`models/decoder.py`. Each decoder is enabled by giving its loss coefficient a
nonzero weight in the `vae:` section:

| Decoder | Predicts from | Coefficient | Default |
| --- | --- | --- | --- |
| `RewardDecoder` | reward from `(z, s_t, a_t)` | `rew_loss_coeff` | `1.0` |
| `StateDecoder` | next state from `(z, s_t, a_t)` | `state_loss_coeff` | `0.0` |
| `TaskDecoder` | task parameters from `z` alone | `task_loss_coeff` | `0.0` |
| `StepAcceptDecoder` | solver accept/reject logit from `(z, s_t, a_t)` | `accept_loss_coeff` | `0.0` |
| `TaskDecoder(task_dim=1)` as end-reward decoder | episode's final reward from `z` alone | `end_reward_loss_coeff` | `0.0` |

### Latent structure (varibad backbone only)

`latent_dim_long > 0` splits `z` into `[z_short, z_long]`:
reward/state/accept decoders read `z_short`; the task and end-reward decoders —
which target task identity rather than per-step dynamics — read `z_long`. With
`latent_dim_long == 0` (default) there is a single latent and every decoder reads
all of it.

Other `vae:` switches:

- `heteroscedastic_recon` — reward/state/task decoders predict `(mean, logvar)`
  and train with a Gaussian NLL instead of MSE. The accept decoder is never
  heteroscedastic (it is already a logit).
- `deterministic_latent` — variational ablation: `z = μ`, no reparameterised
  sampling, and the KL term is forced to zero regardless of `kl_weight`.
- `sequential_kl` — KL against the previous posterior rather than the prior.
  `sequential_kl_propagate_gradient: false` gives a "detached
  gradient" ablation; `sequential_kl_anchor_weight` blends in a fixed `N(0, I)`
  anchor to stop the variance ratchet that detaching otherwise causes.
- `kl_free_bits` — per-dim KL floor (nats) removing the incentive to collapse
  dimensions to the prior.

![Belief encoder-decoder (VAE)](../overleaf/figures/vae_schematic.png)

## Policy

`models/policy.py`'s PPO `ActorCritic`. Actor and critic are **separate MLP
stacks** (`PolicyArchConfig`), not a shared trunk. The action is a single scalar
in `[-1, 1]` (see [environments.md](environments.md)).

`policy.policy_inputs` selects the concatenated input, any subset of
`("state", "z", "task")` that includes at least `"state"` and `"z"`. `"task"` is
excluded by default — belief `z` is the intended channel for task information;
adding `"task"` gives the policy ground truth as an oracle-style upper bound.

Whether the policy consumes a sampled `z` or the concatenated `[μ, log σ²]` is a
property of the backbone (`BackboneSpec.use_latent_sample`); the `varibad`
backbone passes `[μ, log σ²]`.

## Training loop

`training/base_trainer.py` holds the `Trainer` base class and the main loop;
`trainer.py` supplies the belief hooks of the `varibad` backbone
(`_init_belief_model`, `_update_belief_model`, `_belief_checkpoint_fields`).
Each iteration collects trajectories with `training/rollout.py` (vectorized
across `num_envs`), updates the belief model, then updates the policy through
the injected policy algorithm.

Two buffers with different insert/eviction semantics, not one shared buffer:

- `utils/replay_buffer.py` — VAE trajectory replay buffer. Stores raw
  trajectories and re-derives `z` at sample time.
- `utils/policy_buffer.py` — PPO transition buffer, living one iteration.

This distinction matters for **belief staleness**: a stored `z` drifts from what
the current encoder would produce. Bounded and harmless for PPO (one
encoder-gradient-step of lag); unacceptable for any replay-buffer update unless
the buffer stores raw trajectories.
