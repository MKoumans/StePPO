"""Training, rollout, and latent-space plotting utilities."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


def get_series(metrics: list[dict], key: str):
    """Return (iterations, values) of `key` from a list of per-iteration metric dicts."""
    iters, vals = [], []
    for i, m in enumerate(metrics):
        if key in m and m[key] is not None:
            iters.append(i)
            vals.append(m[key])
    return np.array(iters), np.array(vals)


def _single_figure_style(ax, title: str, xlabel: str = "Iteration", ylabel: str = ""):
    SURFACE = "#fcfcfb"
    GRID = "#e1e0d9"
    INK = "#0b0b0b"
    INK_MUTED = "#898781"
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    return SURFACE


def save_hires_diagnostics(metrics: list[dict], output_dir: str):
    """Writes individual high-res diagnostic figures to `output_dir`, one per topic.
    Plots hyperparameter values as actually applied each iteration (post-schedule),
    not the config's nominal values."""
    if not metrics:
        return
    os.makedirs(output_dir, exist_ok=True)

    BLUE = "#2a78d6"
    RED = "#e34948"
    ORANGE = "#d68a2a"

    # 1. Learning rates (PPO + VAE) — the lr actually applied by the optimizer this step.
    ppo_it, ppo_lr = get_series(metrics, "ppo/lr")
    vae_it, vae_lr = get_series(metrics, "vae/lr")
    if len(ppo_it) > 0 or len(vae_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(ax, "Learning Rate (applied)", ylabel="Learning rate")
        if len(ppo_it) > 0:
            ax.plot(ppo_it, ppo_lr, color=BLUE, linewidth=1.3, label="PPO lr")
        if len(vae_it) > 0:
            ax.plot(vae_it, vae_lr, color=RED, linewidth=1.3, label="VAE lr")
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 2. Entropy coefficient (schedule output) vs realised policy entropy.
    ent_it, ent_coeff = get_series(metrics, "ppo/entropy_coeff")
    ent2_it, ent = get_series(metrics, "ppo/entropy")
    if len(ent_it) > 0 or len(ent2_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "Entropy Coefficient (applied)", ylabel="Entropy coefficient"
        )
        lines, labels = [], []
        if len(ent_it) > 0:
            (l1,) = ax.plot(ent_it, ent_coeff, color=BLUE, linewidth=1.3, label="Entropy coeff")
            lines.append(l1)
            labels.append("Entropy coeff")
        if len(ent2_it) > 0:
            ax2 = ax.twinx()
            ax2.set_ylabel("Policy entropy", color=RED, fontsize=9)
            ax2.tick_params(axis="y", labelcolor=RED, labelsize=8)
            (l2,) = ax2.plot(
                ent2_it, ent, color=RED, linewidth=1.0, alpha=0.8, label="Policy entropy"
            )
            lines.append(l2)
            labels.append("Policy entropy")
        ax.legend(lines, labels, fontsize=8, frameon=False, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "entropy_coeff.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 3. Gradient norms (PPO + VAE, pre-clip) — spikes here explain loss spikes elsewhere.
    ppo_g_it, ppo_g = get_series(metrics, "ppo/grad_norm")
    vae_g_it, vae_g = get_series(metrics, "vae/grad_norm")
    if len(ppo_g_it) > 0 or len(vae_g_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(ax, "Gradient Norm (global, pre-clip)", ylabel="Grad norm")
        if len(ppo_g_it) > 0:
            ax.plot(ppo_g_it, ppo_g, color=BLUE, linewidth=1.0, alpha=0.85, label="PPO grad norm")
        if len(vae_g_it) > 0:
            ax.plot(vae_g_it, vae_g, color=RED, linewidth=1.0, alpha=0.85, label="VAE grad norm")
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grad_norms.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 4. Effective KL weight (kl_weight × anneal factor) — the weight actually applied.
    kl_it, kl_w = get_series(metrics, "vae/kl_weight_effective")
    if len(kl_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(ax, "Effective KL Weight (applied)", ylabel="KL weight")
        ax.plot(kl_it, kl_w, color=ORANGE, linewidth=1.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kl_weight_effective.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 4a. Latent posterior variance sum — falling toward ~0 signals posterior collapse.
    var_it, var_sum = get_series(metrics, "vae/latent_var_sum")
    if len(var_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "Latent Posterior Variance Sum", ylabel="Sum of exp(logvar)"
        )
        ax.plot(var_it, var_sum, color=ORANGE, linewidth=1.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_var_sum.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 4b. Pre-clip vs post-clip grad norm; post-clip pinned flat at grad_clip_* means
    # clipping is active on nearly every step, not just rare outliers.
    ppo_pre_it, ppo_pre = get_series(metrics, "ppo/grad_norm")
    _, ppo_post = get_series(metrics, "ppo/grad_norm_post_clip")
    vae_pre_it, vae_pre = get_series(metrics, "vae/grad_norm")
    _, vae_post = get_series(metrics, "vae/grad_norm_post_clip")
    if len(ppo_pre_it) > 0 or len(vae_pre_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "Gradient Norm — Pre-clip vs Post-clip", ylabel="Pre-clip grad norm (log)"
        )
        ax.set_yscale("log")
        lines, labels = [], []
        if len(ppo_pre_it) > 0:
            (line,) = ax.plot(
                ppo_pre_it, ppo_pre, color=BLUE, linewidth=1.0, alpha=0.85, label="PPO pre-clip"
            )
            lines.append(line)
            labels.append("PPO pre-clip")
        if len(vae_pre_it) > 0:
            (line,) = ax.plot(
                vae_pre_it, vae_pre, color=RED, linewidth=1.0, alpha=0.85, label="VAE pre-clip"
            )
            lines.append(line)
            labels.append("VAE pre-clip")

        ax2 = ax.twinx()
        ax2.set_ylabel("Post-clip grad norm (applied)", color=ORANGE, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=8)
        if len(ppo_pre_it) > 0:
            (line,) = ax2.plot(
                ppo_pre_it,
                ppo_post,
                color=ORANGE,
                linewidth=1.0,
                alpha=0.85,
                linestyle="--",
                label="PPO post-clip",
            )
            lines.append(line)
            labels.append("PPO post-clip")
        if len(vae_pre_it) > 0:
            (line,) = ax2.plot(
                vae_pre_it,
                vae_post,
                color="#7a5230",
                linewidth=1.0,
                alpha=0.85,
                linestyle="--",
                label="VAE post-clip",
            )
            lines.append(line)
            labels.append("VAE post-clip")

        ax.legend(lines, labels, fontsize=8, frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grad_norm_pre_post.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 4c. PPO per-term grad norms — actor loss sits near 0 by construction, so loss
    # values aren't a proxy for which term drives the update; plot gradients instead.
    pt_it, g_actor = get_series(metrics, "ppo/grad_norm_actor")
    _, g_value = get_series(metrics, "ppo/grad_norm_value")
    _, g_entropy = get_series(metrics, "ppo/grad_norm_entropy")
    if len(pt_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "PPO Gradient Norm per Term (pre-clip)", ylabel="Grad norm (log)"
        )
        ax.plot(pt_it, g_actor, color=BLUE, linewidth=1.0, alpha=0.85, label="Actor term")
        if len(g_value) > 0:
            ax.plot(
                pt_it,
                g_value,
                color=ORANGE,
                linewidth=1.0,
                alpha=0.85,
                label="Value term (value_coeff-scaled)",
            )
        if len(g_entropy) > 0:
            ax.plot(
                pt_it,
                g_entropy,
                color=RED,
                linewidth=1.0,
                alpha=0.85,
                label="Entropy term (entropy_coeff-scaled)",
            )
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grad_norm_per_term.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 5. PPO loss terms that sum to total_loss (linear axis: entropy can be negative).
    tot_it, tot = get_series(metrics, "ppo/total_loss")
    _, na = get_series(metrics, "ppo/norm_actor_loss")
    _, nv = get_series(metrics, "ppo/norm_value_loss")
    _, ne = get_series(metrics, "ppo/norm_entropy_loss")
    if len(tot_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(ax, "PPO Weighted (optimizer view)", ylabel="Loss term")
        ax.plot(tot_it, tot, color="#0b0b0b", linewidth=1.5, label="Total")
        if len(na) > 0:
            ax.plot(tot_it, na, color=BLUE, linewidth=1.0, alpha=0.85, label="Norm actor")
        if len(nv) > 0:
            ax.plot(
                tot_it, nv, color=ORANGE, linewidth=1.0, alpha=0.85, label="Norm value (weighted)"
            )
        if len(ne) > 0:
            ax.plot(
                tot_it, ne, color=RED, linewidth=1.0, alpha=0.85, label="Norm entropy (unclamped)"
            )
        ax.axhline(0.0, color="#c8c6bd", linewidth=0.8, zorder=0)
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ppo_weighted.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 6. Latent posterior variance sum, tracking encoder variance collapse; right
    # axis is the cumulative sum so a flattening line makes collapse visually obvious.
    var_it, var_sum = get_series(metrics, "vae/latent_var_sum")
    if len(var_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "Latent Posterior Variance (sum over dims)", ylabel="Variance sum"
        )
        ax.plot(var_it, var_sum, color=RED, linewidth=1.3, label="Per-iteration")
        ax2 = ax.twinx()
        ax2.set_ylabel("Cumulative sum (AUC)", color=ORANGE, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=8)
        ax2.plot(
            var_it,
            np.cumsum(var_sum),
            color=ORANGE,
            linewidth=1.3,
            linestyle="--",
            label="Cumulative",
        )
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(
            lines, [line.get_label() for line in lines], fontsize=8, frameon=False, loc="upper left"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latent_variance_sum.png"), dpi=300, facecolor=surface)
        plt.close(fig)

    # 7. Entropy of the accept/reject decoder (0 when accept_loss_coeff == 0).
    ent_it, accept_ent = get_series(metrics, "vae/accept_pred_entropy")
    if len(ent_it) > 0:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        surface = _single_figure_style(
            ax, "Step-Accept Predictive Entropy (given z, state, dt)", ylabel="Entropy (nats)"
        )
        ax.plot(ent_it, accept_ent, color=RED, linewidth=1.3)
        ax.axhline(
            np.log(2),
            color="#c8c6bd",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
            label="ln(2) — maximally uncertain (p=0.5)",
        )
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accept_pred_entropy.png"), dpi=300, facecolor=surface)
        plt.close(fig)


def _write_hyperparam_snapshot(metrics: list[dict], txt_path: str):
    """Writes a plain-text snapshot of the most recently observed value for each
    applied-hyperparameter series, so a human can `cat` the current state
    without opening a plot or the full JSON history."""
    if not metrics:
        return
    keys = [
        "ppo/lr",
        "vae/lr",
        "ppo/entropy_coeff",
        "ppo/entropy",
        "ppo/grad_norm",
        "vae/grad_norm",
        "vae/kl_weight_effective",
    ]
    lines = []
    for key in keys:
        it, vals = get_series(metrics, key)
        if len(it) > 0:
            lines.append(f"{key:<26s} = {vals[-1]:.6g}  (iter {int(it[-1])})")
    if not lines:
        return
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_training_plots(metrics: list[dict], save_path: str):
    """Generates and saves a four-panel training progress plot."""
    if not metrics:
        return

    def series(key):
        return get_series(metrics, key)

    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    # 1. Performance Plot (Train Return with min/max band + Eval Return + Success Rate)
    train_iters, train_returns = series("train/mean_return")
    _, train_min = series("train/min_return")
    _, train_max = series("train/max_return")
    eval_iters, eval_returns = series("eval/mean_return")
    _, eval_success = series("eval/success_rate")

    ax_perf = axs[0]
    has_data = len(train_iters) > 0 or len(eval_iters) > 0

    if has_data:
        ax_perf.set_xlabel("Iteration")
        ax_perf.set_ylabel("Return")
        ax_perf.grid(True, linestyle="--", alpha=0.6)

        if len(train_iters) > 0:
            ax_perf.plot(
                train_iters,
                train_returns,
                color="tab:blue",
                alpha=0.7,
                linewidth=1.0,
                label="Train Return (mean)",
            )
            if len(train_min) == len(train_iters):
                ax_perf.fill_between(
                    train_iters,
                    train_min,
                    train_max,
                    color="tab:blue",
                    alpha=0.15,
                    label="Train Return (min/max)",
                )

        if len(eval_iters) > 0:
            ax_perf.plot(
                eval_iters,
                eval_returns,
                color="tab:blue",
                linewidth=2.5,
                marker="o",
                markersize=4,
                label="Eval Return",
            )

        if len(eval_success) > 0:
            ax_success = ax_perf.twinx()
            ax_success.set_ylabel("Success Rate", color="tab:green")
            ax_success.plot(
                eval_iters,
                eval_success,
                color="tab:green",
                linestyle="--",
                label="Success Rate",
                linewidth=2,
            )
            ax_success.tick_params(axis="y", labelcolor="tab:green")

        ax_perf.legend(loc="lower left", fontsize=8)
        ax_perf.set_title("Training & Evaluation Performance")
    else:
        ax_perf.text(0.5, 0.5, "No Performance Data Yet", ha="center", va="center")
        ax_perf.set_title("Training & Evaluation Performance")

    # 2. VAE Weighted Loss (coeff × EMA-normalised — what optimizer sees)
    ax_w = axs[1]
    w_total_it, w_total = series("vae/total_loss")
    _, w_kl = series("vae/w_kl_loss")
    _, w_rew = series("vae/w_rew_loss")
    _, w_state = series("vae/w_state_loss")
    _, w_task = series("vae/w_task_loss")
    _, w_accept = series("vae/w_accept_loss")

    if len(w_total_it) > 0:
        ax_w.plot(w_total_it, w_total, label="Total", color="black", linewidth=1.5)
        if len(w_kl) > 0:
            ax_w.plot(w_total_it, w_kl, label="KL", color="tab:orange", alpha=0.8)
        if len(w_rew) > 0:
            ax_w.plot(w_total_it, w_rew, label="Reward", color="tab:red", alpha=0.8)
        if len(w_state) > 0:
            ax_w.plot(w_total_it, w_state, label="State", color="tab:purple", alpha=0.8)
        if len(w_task) > 0:
            ax_w.plot(w_total_it, w_task, label="Task", color="tab:cyan", alpha=0.8)
        if len(w_accept) > 0:
            ax_w.plot(w_total_it, w_accept, label="Accept", color="tab:olive", alpha=0.8)
        ax_w.set_xlabel("Iteration")
        ax_w.set_ylabel("Weighted Loss")
        try:
            ax_w.set_yscale("log")
        except ValueError:
            pass
        ax_w.set_title("VAE Weighted (optimizer view)")
        ax_w.legend(fontsize=8)
        ax_w.grid(True, which="both", linestyle="--", alpha=0.4)
    else:
        ax_w.text(0.5, 0.5, "No VAE Data Yet", ha="center", va="center")
        ax_w.set_title("VAE Weighted (optimizer view)")

    # 3. VAE Raw Loss (pre-EMA-norm, no coefficients)
    ax_r = axs[2]
    r_it, r_kl = series("vae/kl_loss")
    _, r_rew = series("vae/rew_loss")
    _, r_state = series("vae/state_loss")
    _, r_task = series("vae/task_loss")
    _, r_accept = series("vae/accept_loss")

    if len(r_it) > 0:
        if len(r_kl) > 0:
            ax_r.plot(r_it, r_kl, label="KL", color="tab:orange", alpha=0.8)
        if len(r_rew) > 0:
            ax_r.plot(r_it, r_rew, label="Reward", color="tab:red", alpha=0.8)
        if len(r_state) > 0:
            ax_r.plot(r_it, r_state, label="State", color="tab:purple", alpha=0.8)
        if len(r_task) > 0:
            ax_r.plot(r_it, r_task, label="Task", color="tab:cyan", alpha=0.8)
        if len(r_accept) > 0:
            ax_r.plot(r_it, r_accept, label="Accept", color="tab:olive", alpha=0.8)
        ax_r.set_xlabel("Iteration")
        ax_r.set_ylabel("Raw Loss")
        try:
            ax_r.set_yscale("log")
        except ValueError:
            pass
        ax_r.set_title("VAE Raw (pre-normalisation)")
        ax_r.legend(fontsize=8)
        ax_r.grid(True, which="both", linestyle="--", alpha=0.4)
    else:
        ax_r.text(0.5, 0.5, "No VAE Data Yet", ha="center", va="center")
        ax_r.set_title("VAE Raw (pre-normalisation)")

    # 4. PPO Loss Plot
    ax_ppo = axs[3]
    ppo_total_it, ppo_total = series("ppo/total_loss")
    _, ppo_actor = series("ppo/actor_loss")
    _, ppo_value = series("ppo/value_loss")
    _, ppo_entropy = series("ppo/entropy")

    if len(ppo_total_it) > 0:
        ax_ppo.plot(ppo_total_it, ppo_total, label="Total Loss", color="black", linewidth=1.5)
        if len(ppo_actor) > 0:
            ax_ppo.plot(ppo_total_it, ppo_actor, label="Actor Loss", color="tab:blue", alpha=0.8)
        if len(ppo_value) > 0:
            ax_ppo.plot(ppo_total_it, ppo_value, label="Value Loss", color="tab:orange", alpha=0.8)

        ax_ppo.set_xlabel("Iteration")
        ax_ppo.set_ylabel("Loss")
        ax_ppo.set_title("PPO Policy Loss")
        ax_ppo.grid(True, linestyle="--", alpha=0.6)

        lines, labels = ax_ppo.get_legend_handles_labels()

        if len(ppo_entropy) > 0:
            ax_entropy = ax_ppo.twinx()
            ax_entropy.plot(
                ppo_total_it, ppo_entropy, label="Policy Entropy", color="tab:red", alpha=0.8
            )
            ax_entropy.set_ylabel("Policy Entropy", color="tab:red")
            ax_entropy.tick_params(axis="y", labelcolor="tab:red")
            entropy_lines, entropy_labels = ax_entropy.get_legend_handles_labels()
            lines, labels = lines + entropy_lines, labels + entropy_labels

        ax_ppo.legend(lines, labels)
    else:
        ax_ppo.text(0.5, 0.5, "No PPO Data Yet\n(Pre-training VAE)", ha="center", va="center")
        ax_ppo.set_title("PPO Policy Loss")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mu_return_history(mu_history: list[tuple[int, list[dict]]], save_path: str):
    """Line plot of mean episode return vs training iteration, one line per μ bin —
    tracks whether reward scale/difficulty differs systematically across μ, and how
    that gap evolves over training. `mu_history` is (iteration, eval_mu_table output) pairs."""
    if not mu_history:
        return

    # Categorical palette, one color per μ bin (reference palette ramp).
    COLORS = [
        "#2a78d6",
        "#e34948",
        "#2aa876",
        "#d68a2a",
        "#8b5cf6",
        "#c2438a",
        "#4a9bd6",
        "#8a8a3c",
    ]

    # Collect the ordered set of μ values seen (preserve first-seen order).
    mus: list[float] = []
    for _, mu_results in mu_history:
        for r in mu_results:
            if r["mu"] not in mus:
                mus.append(r["mu"])

    # Build iteration/mean-return series per μ value.
    series: dict[float, tuple[list[int], list[float]]] = {mu: ([], []) for mu in mus}
    for iteration, mu_results in mu_history:
        for r in mu_results:
            if "returns" not in r or not r["returns"]:
                continue
            iters, vals = series[r["mu"]]
            iters.append(iteration)
            vals.append(float(np.mean(r["returns"])))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    surface = _single_figure_style(ax, "Per-μ return over training", ylabel="Mean episode return")

    for i, mu in enumerate(mus):
        iters, vals = series[mu]
        if not iters:
            continue
        color = COLORS[i % len(COLORS)]
        label = str(int(mu)) if mu == int(mu) else str(mu)
        ax.plot(
            iters, vals, color=color, linewidth=1.6, marker="o", markersize=3, label=f"μ={label}"
        )

    ax.legend(fontsize=8, frameon=False, loc="best", title="μ bin", title_fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=surface)
    plt.close()


def save_mu_return_txt(mu_history: list[tuple[int, list[dict]]], save_path: str):
    """Write plot_mu_return_history's underlying data as a columnar .txt table:
    one row per logged iteration, one column per μ bin (mean return across
    eval repeats). Missing (iteration, μ) pairs are written as 'nan'."""
    if not mu_history:
        return

    mus: list[float] = []
    for _, mu_results in mu_history:
        for r in mu_results:
            if r["mu"] not in mus:
                mus.append(r["mu"])

    rows = []
    for iteration, mu_results in mu_history:
        by_mu = {r["mu"]: r for r in mu_results}
        row = [iteration]
        for mu in mus:
            r = by_mu.get(mu)
            row.append(float(np.mean(r["returns"])) if r and r.get("returns") else float("nan"))
        rows.append(row)

    def _mu_label(mu):
        return f"mu={int(mu)}" if mu == int(mu) else f"mu={mu}"

    header = ["iteration"] + [_mu_label(mu) for mu in mus]
    with open(save_path, "w") as f:
        f.write("# Per-mu return over training\n")
        f.write("  ".join(f"{h:>12s}" for h in header) + "\n")
        for row in rows:
            f.write(f"{row[0]:>12d}" + "".join(f"  {v:>12.4f}" for v in row[1:]) + "\n")


# Colour and label per task split; unknown splits get a muted colour.
_SPLIT_STYLE = {
    "train": ("#2a78d6", "Training distribution"),
    "val": ("#2aa876", "Held-out interior bins"),
    "test": ("#e34948", "Test distribution"),
}
_SPLIT_STYLE_MUTED = "#898781"


def _split_style(name: str) -> tuple[str, str]:
    return _SPLIT_STYLE.get(name, (_SPLIT_STYLE_MUTED, name))


def plot_efficiency_history(
    efficiency_history: list[tuple[int, dict[str, dict]]],
    save_path: str,
):
    """Plot step efficiency relative to PID over training, one line per split.

    Panels: mean of per-task improvement ratios, raw step counts (PID solid,
    policy dashed), and median of ratios.
    """
    if not efficiency_history:
        return

    INK_MUTED = "#898781"

    subsets = _history_subsets(efficiency_history)

    series: dict[str, dict[str, list[float]]] = {
        s: {
            "iters": [],
            "mean": [],
            "std": [],
            "median": [],
            "mean_pid_steps": [],
            "std_pid_steps": [],
            "mean_policy_steps": [],
            "std_policy_steps": [],
        }
        for s in subsets
    }
    for iteration, results in efficiency_history:
        for s in subsets:
            if s not in results:
                continue
            r = results[s]
            d = series[s]
            d["iters"].append(iteration)
            d["mean"].append(r["mean"])
            d["std"].append(r.get("std", 0.0))
            d["median"].append(r.get("median", r["mean"]))
            d["mean_pid_steps"].append(r.get("mean_pid_steps", np.nan))
            d["std_pid_steps"].append(r.get("std_pid_steps", 0.0))
            d["mean_policy_steps"].append(r.get("mean_policy_steps", np.nan))
            d["std_policy_steps"].append(r.get("std_policy_steps", 0.0))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))

    # Panel 1: mean-of-ratios improvement (original metric), with std band.
    surface = _single_figure_style(
        ax1, "Mean of per-task ratios", ylabel="PID-relative step improvement  (pid−policy)/pid"
    )
    for s in subsets:
        d = series[s]
        if not d["iters"]:
            continue
        color, label = _split_style(s)
        iters_arr = np.asarray(d["iters"])
        means_arr = np.asarray(d["mean"])
        stds_arr = np.asarray(d["std"])
        ax1.plot(
            iters_arr, means_arr, color=color, linewidth=1.6, marker="o", markersize=3, label=label
        )
        ax1.fill_between(
            iters_arr,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax1.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle="--")
    ax1.legend(fontsize=8, frameon=False, loc="best")

    # Panel 2: raw step counts, ±1 std over the sampled tasks.
    _single_figure_style(ax2, "Aggregate step counts", ylabel="Mean steps per episode")
    for s in subsets:
        d = series[s]
        if not d["iters"]:
            continue
        color, label = _split_style(s)
        iters_arr = np.asarray(d["iters"])
        pid_m, pid_s = np.asarray(d["mean_pid_steps"]), np.asarray(d["std_pid_steps"])
        pol_m, pol_s = np.asarray(d["mean_policy_steps"]), np.asarray(d["std_policy_steps"])
        ax2.plot(
            iters_arr,
            pid_m,
            color=color,
            linewidth=1.6,
            linestyle="-",
            marker="o",
            markersize=3,
            label=f"{label} — PID",
        )
        ax2.fill_between(
            iters_arr, pid_m - pid_s, pid_m + pid_s, color=color, alpha=0.12, linewidth=0
        )
        ax2.plot(
            iters_arr,
            pol_m,
            color=color,
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{label} — policy",
        )
        ax2.fill_between(
            iters_arr, pol_m - pol_s, pol_m + pol_s, color=color, alpha=0.08, linewidth=0
        )
    ax2.legend(fontsize=7, frameon=False, loc="best")

    # Panel 3: median-of-ratios, robust to small-pid_steps / capped-failure outliers.
    _single_figure_style(
        ax3, "Median of per-task ratios", ylabel="PID-relative step improvement  (pid−policy)/pid"
    )
    for s in subsets:
        d = series[s]
        if not d["iters"]:
            continue
        color, label = _split_style(s)
        iters_arr = np.asarray(d["iters"])
        median_arr = np.asarray(d["median"])
        ax3.plot(
            iters_arr, median_arr, color=color, linewidth=1.6, marker="o", markersize=3, label=label
        )
    ax3.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle="--")
    ax3.legend(fontsize=8, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=surface)
    plt.close()


def _history_subsets(history: list[tuple[int, dict[str, dict]]]) -> list[str]:
    """Every subset name (in first-seen order) appearing anywhere across a
    per-iteration (iteration, {subset: {...}}) history — shared by the
    efficiency and L2-error trend plot/txt writers below, since both walk
    the same history shape."""
    subsets: list[str] = []
    for _, results in history:
        for k in results:
            if k not in subsets:
                subsets.append(k)
    return subsets


def _write_columnar_txt(
    history: list[tuple[int, dict[str, dict]]],
    cols: list[str],
    save_path: str,
) -> None:
    """Write a per-iteration (iteration, {subset: {...}}) history as columnar
    .txt tables, one `# subset: ...` block per subset (in `_history_subsets`
    order), one row per logged iteration that has that subset — shared body
    of save_efficiency_txt/save_l2_error_txt, which differ only in `cols`."""
    if not history:
        return
    subsets = _history_subsets(history)
    with open(save_path, "w") as f:
        for subset in subsets:
            f.write(f"# subset: {subset}\n")
            header = ["iteration"] + cols
            f.write("  ".join(f"{h:>18s}" for h in header) + "\n")
            for iteration, results in history:
                if subset not in results:
                    continue
                r = results[subset]
                row = [r.get(c, float("nan")) for c in cols]
                f.write(f"{iteration:>18d}" + "".join(f"  {v:>18.4f}" for v in row) + "\n")
            f.write("\n")


def save_efficiency_txt(efficiency_history: list[tuple[int, dict[str, dict]]], save_path: str):
    """Write plot_efficiency_history's underlying data as columnar .txt tables,
    one block per task subset (train / val / test, when present), one
    row per logged iteration."""
    _write_columnar_txt(
        efficiency_history,
        [
            "mean",
            "std",
            "median",
            "mean_pid_steps",
            "std_pid_steps",
            "mean_policy_steps",
            "std_policy_steps",
        ],
        save_path,
    )


def plot_l2_error_history(
    l2_error_history: list[tuple[int, dict[str, dict]]],
    save_path: str,
):
    """Plot last-step and time-integrated L2 error (log10) over training, policy vs PID, per split."""
    if not l2_error_history:
        return

    subsets = _history_subsets(l2_error_history)

    series: dict[str, dict[str, list[float]]] = {
        s: {
            "iters": [],
            "policy_err_mean": [],
            "policy_err_std": [],
            "policy_err_integrated_mean": [],
            "policy_err_integrated_std": [],
            "pid_err_mean": [],
            "pid_err_std": [],
            "pid_err_integrated_mean": [],
            "pid_err_integrated_std": [],
        }
        for s in subsets
    }
    for iteration, results in l2_error_history:
        for s in subsets:
            if s not in results:
                continue
            r = results[s]
            d = series[s]
            d["iters"].append(iteration)
            d["policy_err_mean"].append(r["policy_err_mean"])
            d["policy_err_std"].append(r.get("policy_err_std", 0.0))
            d["policy_err_integrated_mean"].append(r["policy_err_integrated_mean"])
            d["policy_err_integrated_std"].append(r.get("policy_err_integrated_std", 0.0))
            d["pid_err_mean"].append(r["pid_err_mean"])
            d["pid_err_std"].append(r.get("pid_err_std", 0.0))
            d["pid_err_integrated_mean"].append(r["pid_err_integrated_mean"])
            d["pid_err_integrated_std"].append(r.get("pid_err_integrated_std", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: last-step L2 error (solid = policy, dashed = PID), shaded by ±1 std.
    surface = _single_figure_style(ax1, "Last-step L2 error", ylabel="log10 relative L2 error")
    for s in subsets:
        d = series[s]
        if not d["iters"]:
            continue
        color, label = _split_style(s)
        iters_arr = np.asarray(d["iters"])
        pol_m, pol_s = np.asarray(d["policy_err_mean"]), np.asarray(d["policy_err_std"])
        pid_m, pid_s = np.asarray(d["pid_err_mean"]), np.asarray(d["pid_err_std"])
        ax1.plot(
            iters_arr,
            pol_m,
            color=color,
            linewidth=1.6,
            linestyle="-",
            marker="o",
            markersize=3,
            label=f"{label} — policy",
        )
        ax1.fill_between(
            iters_arr, pol_m - pol_s, pol_m + pol_s, color=color, alpha=0.12, linewidth=0
        )
        ax1.plot(
            iters_arr,
            pid_m,
            color=color,
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{label} — PID",
        )
        ax1.fill_between(
            iters_arr, pid_m - pid_s, pid_m + pid_s, color=color, alpha=0.08, linewidth=0
        )
    ax1.legend(fontsize=7, frameon=False, loc="best")

    # Panel 2: time-integrated L2 error (solid = policy, dashed = PID), shaded by ±1 std.
    _single_figure_style(
        ax2, "Time-integrated L2 error", ylabel="log10 relative L2 error (integrated)"
    )
    for s in subsets:
        d = series[s]
        if not d["iters"]:
            continue
        color, label = _split_style(s)
        iters_arr = np.asarray(d["iters"])
        pol_m = np.asarray(d["policy_err_integrated_mean"])
        pol_s = np.asarray(d["policy_err_integrated_std"])
        pid_m = np.asarray(d["pid_err_integrated_mean"])
        pid_s = np.asarray(d["pid_err_integrated_std"])
        ax2.plot(
            iters_arr,
            pol_m,
            color=color,
            linewidth=1.6,
            linestyle="-",
            marker="o",
            markersize=3,
            label=f"{label} — policy",
        )
        ax2.fill_between(
            iters_arr, pol_m - pol_s, pol_m + pol_s, color=color, alpha=0.12, linewidth=0
        )
        ax2.plot(
            iters_arr,
            pid_m,
            color=color,
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{label} — PID",
        )
        ax2.fill_between(
            iters_arr, pid_m - pid_s, pid_m + pid_s, color=color, alpha=0.08, linewidth=0
        )
    ax2.legend(fontsize=7, frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=surface)
    plt.close()


def save_l2_error_txt(l2_error_history: list[tuple[int, dict[str, dict]]], save_path: str):
    """Write plot_l2_error_history's underlying data as columnar .txt tables,
    one block per task subset (train / val / test, when present), one
    row per logged iteration."""
    _write_columnar_txt(
        l2_error_history,
        [
            "policy_err_mean",
            "policy_err_std",
            "policy_err_integrated_mean",
            "policy_err_integrated_std",
            "pid_err_mean",
            "pid_err_std",
            "pid_err_integrated_mean",
            "pid_err_integrated_std",
        ],
        save_path,
    )


def plot_mu_advantage_scatter(
    adv_std_per_env: np.ndarray,
    mu_per_env: np.ndarray,
    save_path: str,
    iteration: int,
    n_bins: int = 10,
):
    """Scatter of per-env advantage std vs log(μ) with binned mean overlay — checks
    whether low-μ (high-reward-scale) tasks produce larger advantage magnitudes
    than hard tasks, which would make them dominate PPO gradient updates."""
    if adv_std_per_env is None or len(adv_std_per_env) == 0:
        return

    SURFACE = "#fcfcfb"
    GRID = "#e1e0d9"
    INK = "#0b0b0b"
    INK_MUTED = "#898781"
    BLUE = "#2a78d6"
    RED = "#e34948"  # categorical slot 6 — mean overlay

    log_mu = np.log10(mu_per_env + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Raw scatter — one dot per env
    ax.scatter(log_mu, adv_std_per_env, s=14, color=BLUE, alpha=0.45, linewidths=0, zorder=2)

    # Binned mean ± 1 std overlay
    bin_edges = np.linspace(log_mu.min(), log_mu.max(), n_bins + 1)
    bin_centers, bin_means, bin_stds = [], [], []
    for j in range(n_bins):
        mask = (log_mu >= bin_edges[j]) & (log_mu < bin_edges[j + 1])
        if mask.sum() >= 2:
            bin_centers.append(0.5 * (bin_edges[j] + bin_edges[j + 1]))
            bin_means.append(float(np.mean(adv_std_per_env[mask])))
            bin_stds.append(float(np.std(adv_std_per_env[mask])))

    if bin_centers:
        bc = np.array(bin_centers)
        bm = np.array(bin_means)
        bs = np.array(bin_stds)
        ax.plot(bc, bm, color=RED, linewidth=2.0, zorder=3, label="Binned mean")
        ax.fill_between(bc, bm - bs, bm + bs, color=RED, alpha=0.15, zorder=1)

    # x-axis: show actual μ values at log-spaced ticks
    tick_mus = [1, 2, 5, 10, 20, 50, 100, 200]
    tick_mus = [
        m
        for m in tick_mus
        if np.log10(m) >= log_mu.min() - 0.1 and np.log10(m) <= log_mu.max() + 0.1
    ]
    ax.set_xticks([np.log10(m) for m in tick_mus])
    ax.set_xticklabels([str(m) for m in tick_mus], fontsize=8, color=INK_MUTED)
    ax.set_xlabel("μ (log scale)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Std of GAE advantages (pre-normalisation)", fontsize=9, color=INK_MUTED)
    ax.set_title(
        f"Per-env advantage magnitude vs μ — iter {iteration}\n"
        f"(if left > right: low-μ tasks dominate PPO gradients)",
        fontsize=9,
        color=INK,
    )

    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)

    if bin_centers:
        ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close()


def save_rollout_diagnostic_plot(diagnostics: dict, save_path: str, t_end: float = 1.0):
    """Six-panel diagnostic plot of a single ODE rollout."""
    steps = np.arange(len(diagnostics["reward"]))
    if len(steps) == 0:
        return

    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    # 1. State (y)
    y = diagnostics["y"]
    if y.ndim == 1:
        axs[0].plot(steps, y, linewidth=1)
    else:
        for j in range(y.shape[1]):
            axs[0].plot(steps, y[:, j], linewidth=1, label=f"y{j}")
        axs[0].legend(fontsize=8)
    axs[0].set_ylabel("State")
    axs[0].set_title("ODE State (y)")
    axs[0].grid(True, linestyle="--", alpha=0.4)

    # 2. Progress (t / t_end)
    progress = diagnostics["t"] / t_end
    axs[1].plot(steps, progress, color="tab:blue", linewidth=1.5)
    axs[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axs[1].set_ylabel("t / t_end")
    axs[1].set_title("Progress")
    axs[1].set_ylim(-0.05, 1.1)
    axs[1].grid(True, linestyle="--", alpha=0.4)

    # 3. Step size (dt)
    axs[2].semilogy(steps, np.clip(diagnostics["dt"], 1e-12, None), linewidth=1, color="tab:orange")
    axs[2].set_ylabel("dt")
    axs[2].set_title("Step Size")
    axs[2].grid(True, which="both", linestyle="--", alpha=0.4)

    # 4. Reward
    axs[3].plot(steps, diagnostics["reward"], linewidth=1, color="tab:green")
    axs[3].set_ylabel("Reward")
    axs[3].set_title("Reward")
    axs[3].grid(True, linestyle="--", alpha=0.4)

    # 5. Scaled error
    axs[4].semilogy(
        steps, np.clip(diagnostics["scaled_error"], 1e-12, None), linewidth=1, color="tab:red"
    )
    axs[4].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="accept threshold")
    axs[4].set_ylabel("Scaled Error")
    axs[4].set_title("Scaled Error")
    axs[4].legend(fontsize=8)
    axs[4].grid(True, which="both", linestyle="--", alpha=0.4)

    # 6. Accepted / Rejected strip
    keep = diagnostics["keep_step"].astype(bool)
    colors = np.where(keep, "tab:green", "tab:red")
    axs[5].bar(steps, 1, width=1.0, color=colors, edgecolor="none")
    axs[5].set_yticks([])
    axs[5].set_title("Accepted (green) / Rejected (red)")
    axs[5].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


_LATENT_SCATTER_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "orange_blue", ["#E8873A", "#2a78d6"]
)


def _latent_scatter_norm(task_values: np.ndarray):
    """Color normalisation for task values: log scale when they span >1 decade."""
    is_log = bool(np.all(task_values > 0)) and (
        task_values.max() / max(task_values.min(), 1e-12) > 10
    )
    norm = (
        matplotlib.colors.LogNorm(vmin=task_values.min(), vmax=task_values.max())
        if is_log
        else matplotlib.colors.Normalize(vmin=task_values.min(), vmax=task_values.max())
    )
    return norm, is_log


def _render_latent_scatter_frame(
    x: np.ndarray,
    y: np.ndarray,
    task_values: np.ndarray,
    iteration: int,
    dims: tuple[int, int],
    png_path: str,
    norm,
    is_log: bool,
    axis_limit: float | None = None,
):
    """Renders one latent scatter frame. When `axis_limit` is given, both axes are
    fixed to the symmetric range [-axis_limit, axis_limit] (used for GIF frames so
    every frame shares the same square grid)."""
    SURFACE = "#fcfcfb"
    GRID = "#e1e0d9"
    INK = "#0b0b0b"
    INK_MUTED = "#898781"

    dim_x, dim_y = dims

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    sc = ax.scatter(
        x,
        y,
        c=task_values,
        cmap=_LATENT_SCATTER_CMAP,
        norm=norm,
        s=18,
        alpha=0.75,
        linewidths=0,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(
        "Task parameter λ" + (" (log scale)" if is_log else ""), fontsize=9, color=INK_MUTED
    )
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)

    ax.set_xlabel(f"z[{dim_x}]", fontsize=9, color=INK_MUTED)
    ax.set_ylabel(f"z[{dim_y}]", fontsize=9, color=INK_MUTED)
    ax.set_title(f"Latent space (belief μ) — iter {iteration}", fontsize=11, color=INK)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)

    if axis_limit is not None:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(png_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def save_latent_scatter_snapshot(
    z: np.ndarray,
    task_values: np.ndarray,
    iteration: int,
    dims: tuple[int, int],
    png_path: str,
    csv_path: str,
):
    """Scatter of belief_mu[dims] colored by true task value; appends samples to a
    running CSV table (one row per sample per iteration) for later GIF assembly."""
    import csv

    dim_x, dim_y = dims
    norm, is_log = _latent_scatter_norm(task_values)
    _render_latent_scatter_frame(
        z[:, dim_x],
        z[:, dim_y],
        task_values,
        iteration,
        dims,
        png_path,
        norm,
        is_log,
    )

    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["iteration", "sample_idx", "task_value"] + [f"z{i}" for i in range(z.shape[1])]
            )
        for i in range(z.shape[0]):
            writer.writerow([iteration, i, float(task_values[i])] + [float(v) for v in z[i]])


def assemble_latent_scatter_gif(
    png_dir: str,
    gif_path: str,
    fps: int = 4,
    csv_path: str | None = None,
    dims: tuple[int, int] = (0, 1),
):
    """Assembles the latent-scatter GIF showing latent distribution shift over training.
    When `csv_path` exists, frames are re-rendered with a shared square axis range
    [-L, L] (L = smallest whole number > max |z|) and global colorbar; otherwise
    falls back to stitching the per-iteration autoscaled `iter_*.png` snapshots."""
    import glob

    from PIL import Image

    paths = sorted(glob.glob(os.path.join(png_dir, "iter_*.png")))

    if csv_path is not None and os.path.exists(csv_path):
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        data = np.atleast_1d(data)
        dim_x, dim_y = dims
        x_all = data[f"z{dim_x}"]
        y_all = data[f"z{dim_y}"]
        task_all = data["task_value"]
        iters = data["iteration"].astype(int)

        max_abs = float(max(np.abs(x_all).max(), np.abs(y_all).max()))
        axis_limit = float(np.floor(max_abs)) + 1.0  # whole number > largest |z|
        norm, is_log = _latent_scatter_norm(task_all)

        frame_dir = os.path.join(png_dir, "gif_frames")
        os.makedirs(frame_dir, exist_ok=True)
        paths = []
        for it in np.unique(iters):
            mask = iters == it
            frame_path = os.path.join(frame_dir, f"iter_{it:07d}.png")
            _render_latent_scatter_frame(
                x_all[mask],
                y_all[mask],
                task_all[mask],
                int(it),
                dims,
                frame_path,
                norm,
                is_log,
                axis_limit=axis_limit,
            )
            paths.append(frame_path)
        print(f"[+] Re-rendered {len(paths)} GIF frames with fixed range ±{axis_limit:g}.")

    if not paths:
        print(f"[-] No latent scatter snapshots found in '{png_dir}' — skipping GIF.")
        return

    frames = [Image.open(p).convert("RGB") for p in paths]
    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[+] Saved latent scatter GIF ({len(frames)} frames) to: {gif_path}")
