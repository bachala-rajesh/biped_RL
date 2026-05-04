# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SIMPLE SCRIPT: Test the copycat robot

This script:
1. Loads your trained copycat
2. Puts it in the simulator
3. Sees if it can walk like the expert
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the copycat robot.")
parser.add_argument("--task", type=str, default="biped_walk_flat_play", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="How many robots")
parser.add_argument("--policy", type=str, default="behaviour_cloning/bc_policy.pt", help="Path to trained BC policy")
parser.add_argument("--steps", type=int, default=1000, help="How many steps to run")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import biped.tasks  # noqa: F401


class CopycatRobot(nn.Module):
    """Same network architecture as training."""
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.net(x)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Run the copycat robot."""
    
    # Load the copycat
    print(f"[INFO] Loading copycat from: {args_cli.policy}")
    checkpoint = torch.load(args_cli.policy)
    
    copycat = CopycatRobot(checkpoint["obs_size"], checkpoint["action_size"])
    copycat.load_state_dict(checkpoint["model"])
    copycat.eval()
    
    # Create environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    device = env.unwrapped.device
    copycat = copycat.to(device)
    
    print(f"[INFO] Running copycat for {args_cli.steps} steps...")
    print("[INFO] Watch and see if it walks like the expert!\n")
    
    obs = env.get_observations()
    
    for step in range(args_cli.steps):
        with torch.inference_mode():
            # Copycat decides what to do
            actions = copycat(obs)
            
            # Do it
            obs, rewards, dones, infos = env.step(actions)
            
            # Print how well it's doing
            if (step + 1) % 100 == 0:
                avg_reward = rewards.mean().item()
                alive = (~dones).sum().item()
                print(f"  Step {step + 1}: Avg Reward = {avg_reward:.3f}, Robots alive = {alive}/{args_cli.num_envs}")
    
    print("\n[INFO] Done! Close the window to exit.")
    
    # Keep window open
    while simulation_app.is_running():
        env.step(actions)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
