# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SIMPLE SCRIPT: Collect data from your trained robot

This script:
1. Loads your trained robot
2. Watches it walk
3. Writes down: "When robot saw X, it did Y"
4. Saves to a file for training a copycat robot
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect data from trained robot.")
parser.add_argument(
    "--task",
    type=str,
    default="biped_walk_flat_play",
    help="Task name (use the _play version)",
)
parser.add_argument(
    "--num_envs", type=int, default=64, help="How many robots to run at same time"
)
parser.add_argument("--steps", type=int, default=1000, help="How many steps to record")
parser.add_argument(
    "--output",
    type=str,
    default="behaviour_cloning/expert_data.pt",
    help="Where to save the data",
)
parser.add_argument(
    "--load_run", type=str, required=True, help="Name of your trained run folder"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import biped.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Collect data from trained policy."""

    # Set agent config for evaluation (no training resume needed)
    agent_cfg.resume = False

    # Set up the environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    # Disable events (external forces, randomizations) for clean data collection
    # Only disable specific events, NOT all events (we need reset events!)
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
            print("[INFO] Disabled robot push events")
        if hasattr(env_cfg.events, "add_base_mass"):
            env_cfg.events.add_base_mass = None
            print("[INFO] Disabled base mass randomization")

    # Also disable command randomization if present
    if hasattr(env_cfg, "commands"):
        commands_cfg = env_cfg.commands
        # Disable resampling for base_velocity command
        if hasattr(commands_cfg, "base_velocity") and hasattr(
            commands_cfg.base_velocity, "resampling_time_range"
        ):
            commands_cfg.base_velocity.resampling_time_range = (1e9, 1e9)
            print("[INFO] Disabled base_velocity command resampling")
        # Disable resampling for gait_command if present
        if hasattr(commands_cfg, "gait_command") and hasattr(
            commands_cfg.gait_command, "resampling_time_range"
        ):
            commands_cfg.gait_command.resampling_time_range = (1e9, 1e9)
            print("[INFO] Disabled gait_command resampling")

    # Find your trained robot
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, "model_.*.pt")

    print(f"[INFO] Loading robot from: {resume_path}")

    # Create the robot world
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load your trained robot brain
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Storage for what we see
    all_observations = []
    all_actions = []

    print(f"[INFO] Recording {args_cli.steps} steps from {args_cli.num_envs} robots...")
    print(
        f"[INFO] This will collect ~{args_cli.steps * args_cli.num_envs} total examples"
    )

    # Start watching
    obs = env.get_observations()

    for step in range(args_cli.steps):
        # Watch what the robot does
        with torch.inference_mode():
            actions = policy(obs)

            # SAVE: What robot saw and what it did
            all_observations.append(obs.cpu().clone())
            all_actions.append(actions.cpu().clone())

            # Continue to next step
            obs, _, dones, _ = env.step(actions)

        # Progress bar
        if (step + 1) % 100 == 0:
            print(f"  Recorded: {step + 1}/{args_cli.steps} steps")

    # Save everything to file
    data = {
        "observations": torch.cat(all_observations, dim=0),  # All robot sees
        "actions": torch.cat(all_actions, dim=0),  # All robot does
    }

    # Make sure output directory exists
    os.makedirs(
        os.path.dirname(args_cli.output) if os.path.dirname(args_cli.output) else ".",
        exist_ok=True,
    )
    torch.save(data, args_cli.output)
    print(
        f"\n[SUCCESS] Saved {len(data['observations'])} examples to: {args_cli.output}"
    )
    print(f"  File size: {os.path.getsize(args_cli.output) / 1024 / 1024:.1f} MB")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
