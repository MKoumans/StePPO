import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from steppo.configs.base_config import PPOConfig, TrainConfig, VAEConfig
from steppo.training.trainer import VariBADTrainer
from steppo.utils.device import get_device_mesh, setup_devices, shard_array, shard_tree
from steppo.utils.logger import MetricsLogger
from steppo.utils.plotting import (
    plot_efficiency_history,
    plot_l2_error_history,
    save_efficiency_txt,
    save_l2_error_txt,
    save_training_plots,
)


class TestLoggingExportAndDevice(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_argv = list(sys.argv)
        self.orig_environ = dict(os.environ)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        sys.argv = self.orig_argv
        os.environ.clear()
        os.environ.update(self.orig_environ)

    def test_save_training_plots(self):
        # Create dummy metrics matching VAE pretraining, joint training, and evaluations
        dummy_metrics = []
        for i in range(30):
            m = {
                "vae/total_loss": 2.0 / (i + 1),
                "vae/kl_loss": 0.5 / (i + 1),
                "vae/rew_loss": 1.5 / (i + 1),
                "vae/state_loss": 0.1 / (i + 1),
            }
            if i >= 10:
                m.update(
                    {
                        "ppo/total_loss": 1.0 - 0.01 * i,
                        "ppo/actor_loss": -0.05,
                        "ppo/value_loss": 0.8 - 0.01 * i,
                        "ppo/entropy": 0.5 + 0.002 * i,
                    }
                )
            if i % 10 == 0:
                m.update(
                    {
                        "eval/mean_return": -10.0 + 0.2 * i,
                        "eval/success_rate": 0.01 * i,
                        "eval/mean_episode_length": 50.0 - 0.1 * i,
                    }
                )
            dummy_metrics.append(m)

        plot_path = os.path.join(self.test_dir, "test_plot.png")
        save_training_plots(dummy_metrics, plot_path)

        # Verify file exists and is non-empty
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

    def test_plot_efficiency_history_handles_val_subset(self):
        """plot_efficiency_history/save_efficiency_txt (src/steppo/utils/plotting.py)
        must discover a 'val' subset dynamically like train/test/noise, without a
        real training run — this is the wiring eval_efficiency_metric's val_bins
        support relies on, previously verified only via a manual smoketest."""

        def _row(mean):
            return {
                "mean": mean,
                "std": 0.01,
                "median": mean,
                "mean_pid_steps": 100.0,
                "mean_policy_steps": 90.0,
            }

        efficiency_history = [
            (
                i,
                {
                    "train": _row(0.1 * i),
                    "test": _row(0.05 * i),
                    "noise": _row(0.02 * i),
                    "val": _row(0.08 * i),
                },
            )
            for i in range(3)
        ]

        plot_path = os.path.join(self.test_dir, "efficiency.png")
        plot_efficiency_history(efficiency_history, plot_path)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

        txt_path = os.path.join(self.test_dir, "efficiency.txt")
        save_efficiency_txt(efficiency_history, txt_path)
        with open(txt_path) as f:
            content = f.read()
        self.assertIn("# subset: val", content)

    def _l2_row(self, scale):
        return {
            "policy_err_mean": 0.1 * scale,
            "policy_err_std": 0.01,
            "policy_err_integrated_mean": 0.2 * scale,
            "policy_err_integrated_std": 0.02,
            "pid_err_mean": 0.15 * scale,
            "pid_err_std": 0.015,
            "pid_err_integrated_mean": 0.25 * scale,
            "pid_err_integrated_std": 0.025,
        }

    def test_plot_l2_error_history_single_subset(self):
        """plot_l2_error_history writes a non-empty plot for a synthetic 2-iteration,
        single-subset history (the L2-error trend counterpart to
        plot_efficiency_history, see src/steppo/utils/plotting.py)."""
        l2_error_history = [(i, {"train": self._l2_row(i + 1)}) for i in range(2)]

        plot_path = os.path.join(self.test_dir, "l2_error.png")
        plot_l2_error_history(l2_error_history, plot_path)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

    def test_plot_l2_error_history_handles_val_subset(self):
        """Mirrors test_plot_efficiency_history_handles_val_subset: must discover a
        'val' subset dynamically like train/test/noise, and must not error when an
        iteration is missing the 'val' subset entirely."""
        l2_error_history = [
            (0, {"train": self._l2_row(1), "test": self._l2_row(1), "val": self._l2_row(1)}),
            (1, {"train": self._l2_row(2), "test": self._l2_row(2)}),  # val missing here
        ]

        plot_path = os.path.join(self.test_dir, "l2_error.png")
        plot_l2_error_history(l2_error_history, plot_path)
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 0)

        txt_path = os.path.join(self.test_dir, "l2_error.txt")
        save_l2_error_txt(l2_error_history, txt_path)
        with open(txt_path) as f:
            content = f.read()
        self.assertIn("# subset: val", content)

    def test_save_l2_error_txt_header_columns(self):
        """save_l2_error_txt's header row must contain all 8 documented columns
        for a subset that's present."""
        l2_error_history = [(i, {"train": self._l2_row(i + 1)}) for i in range(2)]

        txt_path = os.path.join(self.test_dir, "l2_error.txt")
        save_l2_error_txt(l2_error_history, txt_path)
        with open(txt_path) as f:
            content = f.read()

        for col in [
            "policy_err_mean",
            "policy_err_std",
            "policy_err_integrated_mean",
            "policy_err_integrated_std",
            "pid_err_mean",
            "pid_err_std",
            "pid_err_integrated_mean",
            "pid_err_integrated_std",
        ]:
            self.assertIn(col, content)

    def test_l2_error_writers_noop_on_empty_history(self):
        """Both L2-error writers must be no-ops (no error, no file written) when
        given an empty history list, mirroring plot_efficiency_history/
        save_efficiency_txt's empty-history handling."""
        plot_path = os.path.join(self.test_dir, "l2_error_empty.png")
        txt_path = os.path.join(self.test_dir, "l2_error_empty.txt")

        plot_l2_error_history([], plot_path)
        save_l2_error_txt([], txt_path)

        self.assertFalse(os.path.exists(plot_path))
        self.assertFalse(os.path.exists(txt_path))

    def test_metrics_logger_running_avg(self):
        logger = MetricsLogger(use_wandb=False)

        # Log some values
        logger.update_history({"loss": 1.0})
        logger.update_history({"loss": 2.0})
        logger.update_history({"loss": 3.0})

        self.assertAlmostEqual(logger.get_running_avg("loss"), 2.0)
        self.assertAlmostEqual(logger.get_running_avg("missing_key", 9.9), 9.9)

    def test_metrics_logger_print_and_eta(self):
        logger = MetricsLogger(use_wandb=False)

        # Test printing does not raise exceptions
        metrics = {
            "vae/total_loss": 1.0,
            "vae/kl_loss": 0.2,
            "vae/rew_loss": 0.8,
            "ppo/total_loss": 0.5,
            "ppo/actor_loss": -0.1,
            "ppo/value_loss": 0.6,
            "ppo/entropy": 0.3,
        }

        # Ensure log_training runs successfully
        logger.log_training(step=10, total_steps=100, metrics=metrics, step_time=0.5)
        logger.log_evaluation(step=10, metrics={"eval/mean_return": 1.5, "eval/success_rate": 0.8})

    def test_setup_devices_cli_arg(self):
        sys.argv = ["run.py", "--gpus", "0,1"]
        setup_devices()
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1")
        self.assertNotIn("--gpus", sys.argv)

    def test_setup_devices_env_var(self):
        sys.argv = ["run.py"]
        os.environ["GPUS"] = "2,3"
        setup_devices()
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2,3")

    def test_setup_devices_cpu_fallback(self):
        sys.argv = ["run.py", "--gpus", "cpu"]
        setup_devices()
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def test_sharding_fallback_single_device(self):
        # Sharding with None or 1-device mesh should return the array unchanged
        arr = jnp.arange(10)
        sharded_arr = shard_array(arr, None)
        self.assertTrue(jnp.array_equal(arr, sharded_arr))

        tree = {"a": arr, "b": {"c": arr}}
        sharded_tree = shard_tree(tree, None)
        self.assertTrue(jnp.array_equal(tree["a"], sharded_tree["a"]))
        self.assertTrue(jnp.array_equal(tree["b"]["c"], sharded_tree["b"]["c"]))

    def test_device_mesh_creation(self):
        mesh = get_device_mesh()
        self.assertIsNotNone(mesh)
        self.assertIn("data", mesh.axis_names)

    def test_unsafe_hash_config(self):
        # Dataclasses with unsafe_hash=True should be hashable
        vae_cfg = VAEConfig()
        ppo_cfg = PPOConfig()
        train_cfg = TrainConfig()

        self.assertIsNotNone(hash(vae_cfg))
        self.assertIsNotNone(hash(ppo_cfg))
        self.assertIsNotNone(hash(train_cfg))

    def test_run_uid_generation(self):
        cfg = TrainConfig()
        self.assertEqual(len(cfg.run_uid), 8)  # 8-char uuid4 hex
        int(cfg.run_uid, 16)  # raises if not hex

        # Check customized run_uid
        cfg2 = TrainConfig(run_uid="custom_run", run_date="20260101")
        self.assertEqual(cfg2.run_uid, "custom_run")

        # Default paths also include env family and system.
        self.assertEqual(
            cfg2.checkpoint_path,
            os.path.abspath(
                os.path.join(
                    "outputs",
                    "runs",
                    "20260101",
                    "ode",
                    "van_der_pol",
                    "custom_run",
                    "checkpoints",
                )
            ),
        )
        self.assertEqual(
            cfg2.output_path,
            os.path.abspath(
                os.path.join(
                    "outputs",
                    "runs",
                    "20260101",
                    "ode",
                    "van_der_pol",
                    "custom_run",
                    "outputs",
                )
            ),
        )

        # Explicit roots override the default hierarchy and append only run_uid.
        cfg3 = TrainConfig(
            run_uid="e1",
            run_date="20260101",
            checkpoints_dir="/tmp/exp/checkpoints",
            outputs_dir="/tmp/exp/outputs",
        )
        self.assertEqual(
            cfg3.checkpoint_path,
            os.path.join("/tmp/exp/checkpoints", "e1"),
        )
        self.assertEqual(
            cfg3.output_path,
            os.path.join("/tmp/exp/outputs", "e1"),
        )

    def test_checkpointing_save_and_restore(self):
        from steppo.training.ppo_trainer import PPOTrainState
        from steppo.training.vae_trainer import VAETrainState

        cfg = TrainConfig(run_uid="test_ckpt_run", checkpoints_dir=self.test_dir)

        # Dummy vae and ppo states
        vae_state = VAETrainState(
            graphdef="vae_graph",
            params={"w": jnp.array([1.0, 2.0])},
            opt_state={"opt": jnp.array(0)},
            step=12,
            ema_rew=jnp.float32(0.0),
            ema_state=jnp.float32(0.0),
            ema_kl=jnp.float32(0.0),
            ema_task=jnp.float32(0.0),
            ema_accept=jnp.float32(0.0),
            ema_end_reward=jnp.float32(0.0),
        )
        ppo_state = PPOTrainState(
            graphdef="ppo_graph",
            params={"w": jnp.array([3.0, 4.0])},
            opt_state={"opt": jnp.array(1)},
            step=34,
            ema_actor=jnp.float32(0.0),
            ema_value=jnp.float32(0.0),
        )

        mock_env = MagicMock()
        mock_policy = MagicMock()
        mock_vae = MagicMock()

        trainer = VariBADTrainer(
            config=cfg, vae=mock_vae, policy=mock_policy, env=mock_env, env_params=MagicMock()
        )
        trainer._control_envelopes = {}

        # Save checkpoint
        ckpt_dir = trainer.save_checkpoint(vae_state, ppo_state, iteration=5)
        self.assertTrue(os.path.exists(ckpt_dir))

        # Restore checkpoint
        restored = trainer.load_checkpoint(ckpt_dir)
        self.assertEqual(restored["iteration"], 5)
        self.assertEqual(restored["vae_step"], 12)
        self.assertEqual(restored["ppo_step"], 34)
        self.assertTrue(np.array_equal(restored["vae_params"]["w"], np.array([1.0, 2.0])))
        self.assertTrue(np.array_equal(restored["ppo_params"]["w"], np.array([3.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
