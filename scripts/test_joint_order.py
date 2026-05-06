# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify that observation, action, and articulation joint orders match."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test joint ordering in the biped environment.")
parser.add_argument("--task", type=str, default="biped_walk_flat", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import biped.tasks  # noqa: F401


def main():
    # parse env configuration
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    robot = env.unwrapped.scene["robot"]
    action_mgr = env.unwrapped.action_manager
    obs_mgr = env.unwrapped.observation_manager

    desired_order = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
    ]

    print("=" * 60)
    print("ARTICULATION INTERNAL JOINT ORDER (PhysX DOF order)")
    print("=" * 60)
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")

    print("\n" + "=" * 60)
    print("ACTION MANAGER JOINT ORDER")
    print("=" * 60)
    action_term = action_mgr._terms["joint_pos"]
    action_joint_names = action_term.cfg.joint_names
    for i, name in enumerate(action_joint_names):
        print(f"  [{i}] {name}")

    print("\n" + "=" * 60)
    print("OBSERVATION MANAGER - Policy group terms")
    print("=" * 60)
    policy_terms = obs_mgr.active_terms["policy"]
    for term_name in policy_terms:
        print(f"  - {term_name}")

    # Find joint_pos and joint_vel observation configs
    print("\n" + "=" * 60)
    print("OBSERVATION MANAGER - 'joint_pos' resolved asset_cfg")
    print("=" * 60)
    idx_pos = policy_terms.index("joint_pos")
    obs_cfg_pos = obs_mgr._group_obs_term_cfgs["policy"][idx_pos]
    if "asset_cfg" in obs_cfg_pos.params:
        obs_joint_names_pos = obs_cfg_pos.params["asset_cfg"].joint_names
        for i, name in enumerate(obs_joint_names_pos):
            print(f"  [{i}] {name}")
    else:
        print("  WARNING: no asset_cfg in params -> using articulation internal order!")
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i}] {name}")

    print("\n" + "=" * 60)
    print("OBSERVATION MANAGER - 'joint_vel' resolved asset_cfg")
    print("=" * 60)
    idx_vel = policy_terms.index("joint_vel")
    obs_cfg_vel = obs_mgr._group_obs_term_cfgs["policy"][idx_vel]
    if "asset_cfg" in obs_cfg_vel.params:
        obs_joint_names_vel = obs_cfg_vel.params["asset_cfg"].joint_names
        for i, name in enumerate(obs_joint_names_vel):
            print(f"  [{i}] {name}")
    else:
        print("  WARNING: no asset_cfg in params -> using articulation internal order!")
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i}] {name}")

    # --- Functional action test ---
    print("\n" + "=" * 60)
    print("FUNCTIONAL ACTION ORDER TEST")
    print("=" * 60)

    env.reset()
    # Send a one-hot action: [1.0, 0, 0, 0, 0, 0]
    actions = torch.zeros((args_cli.num_envs, action_mgr.total_action_dim), device=robot.device)
    actions[0, 0] = 1.0
    env.step(actions)

    # Read the processed action target directly from the action term
    action_term = action_mgr._terms["joint_pos"]
    # The action term stores processed actions (after scaling and offset)
    # In Isaac Lab, the action term applies targets to the articulation.
    # We can inspect the articulation's joint target or just check the action mapping.
    # A cleaner way: check what the action manager's full action vector looks like.
    print(f"Sent action        : {actions[0].cpu().numpy()}")
    print(f"Processed action   : {action_mgr.action[0].cpu().numpy()}")

    # The action term cfg tells us the order; we already printed it above.
    # To functionally verify, we compare the action term's joint_names with desired_order.
    action_joint_names = action_term.cfg.joint_names
    action_match = (action_joint_names == desired_order)
    print(f"Action term names match desired order: {'PASS ✅' if action_match else 'FAIL ❌'}")

    # --- Functional observation test ---
    print("\n" + "=" * 60)
    print("FUNCTIONAL OBSERVATION ORDER TEST")
    print("=" * 60)

    env.reset()
    # Set joint positions manually to a known pattern in the desired order
    target_pos = torch.zeros((args_cli.num_envs, robot.num_joints), device=robot.device)
    for i, name in enumerate(desired_order):
        internal_idx = robot.joint_names.index(name)
        target_pos[0, internal_idx] = 0.1 * (i + 1)  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    robot.write_joint_state_to_sim(target_pos, robot.data.default_joint_vel.clone())
    robot.update(0.0)
    env.unwrapped.scene.update(0.0)

    # Compute observations
    obs_dict = obs_mgr.compute()
    policy_obs = obs_dict["policy"][0]

    # With history_length=5 and flatten_history_dim=True, obs is flat.
    # We need to figure out the per-step dim.
    # The obs_mgr knows the dims per term.
    group_dim = obs_mgr.group_obs_dim["policy"]
    # group_dim may be a tuple like (160,)
    if isinstance(group_dim, tuple):
        group_dim = group_dim[0]
    if policy_obs.numel() == group_dim:
        latest_obs = policy_obs
    else:
        # flattened history: total = group_dim * history_length
        latest_obs = policy_obs[-group_dim:]

    # Parse the observation terms to find exact indices
    # Terms order: base_ang_vel(3), proj_gravity(3), vel_command(3), joint_pos(6), joint_vel(6), last_action(6), gait_phase(2), gait_command(3)
    term_dims = obs_mgr.group_obs_term_dim["policy"]
    term_names = obs_mgr.active_terms["policy"]
    cursor = 0
    joint_pos_slice = None
    for name, dim in zip(term_names, term_dims):
        dim_val = dim[0] if isinstance(dim, tuple) else dim
        if name == "joint_pos":
            joint_pos_slice = slice(cursor, cursor + dim_val)
        cursor += dim_val

    if joint_pos_slice is None:
        print("ERROR: could not find joint_pos in observation terms")
    else:
        joint_pos_obs = latest_obs[joint_pos_slice]
        print(f"Set robot joints (desired order) : {[0.1*(i+1) for i in range(6)]}")
        print(f"Observed joint_pos values        : {joint_pos_obs.cpu().numpy()}")

        # Expected relative positions in the observation order
        if "asset_cfg" in obs_cfg_pos.params:
            expected_names = obs_cfg_pos.params["asset_cfg"].joint_names
        else:
            expected_names = robot.joint_names

        expected = []
        for name in expected_names:
            internal_idx = robot.joint_names.index(name)
            expected.append(target_pos[0, internal_idx].item() - default_pos[internal_idx].item())

        expected_tensor = torch.tensor(expected, device=robot.device, dtype=joint_pos_obs.dtype)
        obs_match = torch.allclose(joint_pos_obs, expected_tensor, atol=1e-3)
        print(f"Observation order test: {'PASS ✅' if obs_match else 'FAIL ❌'}")

    print("\n" + "=" * 60)
    if action_match and (obs_match if joint_pos_slice else False):
        print("ALL ORDER TESTS PASSED")
    else:
        print("ORDER MISMATCH DETECTED — review output above")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
