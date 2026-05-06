"""Script for sim2sim deployment of a trained RLpolicy in Mujoco environment"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import torch
from utils import HistoryBuffer, get_mujoco_data


# command velocity
cmd_vel = np.array([0.0, 0.0, 0.0])

relative_policy_path = (
    "logs/rsl_rl/bipedal_locomotion/2026-05-04_13-44-30_flat/exported/policy.pt"
)



from sim_config import Sim2simCfg


def get_observation(
    model, data, last_actions, gait_time, cmd_vel, initial_joint_pos, obs_gait_command
):
    obs_proj_gravity, _, ang_vel, joints_pos, joints_vel = get_mujoco_data(
        model, data, list(Sim2simCfg.robot_config.joint_names.keys())
    )

    # calculate gait phase
    gait_phase_val = (gait_time * Sim2simCfg.robot_config.gait_freq) % 1.0
    obs_gait_phase_sin_cos = np.array(
        [
            np.sin(2 * np.pi * gait_phase_val),
            np.cos(2 * np.pi * gait_phase_val),
        ],
        dtype=np.float32,
    )

    # apply scaling to observations  
    obs_ang_vel = ang_vel * Sim2simCfg.robot_config.ang_vel_scale
    relative_joint_pos = joints_pos - initial_joint_pos
    obs_joint_pos = relative_joint_pos * Sim2simCfg.robot_config.dof_pos_scale
    obs_joint_vel = joints_vel * Sim2simCfg.robot_config.dof_vel_scale
    obs_cmd = cmd_vel * np.array(
        [
            Sim2simCfg.robot_config.lin_vel_scale,
            Sim2simCfg.robot_config.lin_vel_scale,
            Sim2simCfg.robot_config.ang_vel_scale,
        ],
        dtype=np.float32,
    )
    obs_last_actions = last_actions

    # form the observation vector as a list of numpy arrays
    current_obs = [
        obs_ang_vel.reshape(-1),  # [:,3]
        obs_proj_gravity.reshape(-1),  # [:,3]
        obs_cmd.reshape(-1),  # [:,3]
        obs_joint_pos.reshape(-1),  # [:,6]
        obs_joint_vel.reshape(-1),  # [:,6]
        obs_last_actions.reshape(-1),  # [:,6]
        obs_gait_phase_sin_cos.reshape(-1),  # [:,2]
        obs_gait_command.reshape(-1),  # [:,3]
    ]

    fall_status = False
    if abs(obs_proj_gravity[0]) > 0.90:
        fall_status = True

    return current_obs, fall_status


def run_mujoco(rl_model_path, robot_model_path, cmd_vel):
    # load policy
    print(f"Loading Policy: {rl_model_path}")
    policy = torch.jit.load(rl_model_path)
    policy.eval()

    # load model
    print(f"Loading Model: {robot_model_path}")
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)

    # set gains
    model.actuator_gainprm[:, 0] = Sim2simCfg.robot_config.stiffness_gain  # Stiffness
    model.actuator_biasprm[:, 2] = Sim2simCfg.robot_config.damping_gain  # Damping

    # init history buffer
    history_buffer = HistoryBuffer(
        obs_history_len=Sim2simCfg.robot_config.obs_history_len,
        num_obs_terms=Sim2simCfg.robot_config.num_obs_terms,
    )

    # loop variables
    step_counter = 0
    last_actions = np.zeros(6, dtype=np.float32)
    gait_time_accumulator = 0.0

    #  time related variables
    start_time = time.time()
    real_start_time = time.time()
    warmup_delay = 0.10  # Wait few seconds before turning on Policy

    # initial joint positions
    initial_joint_pos = np.array(
        [
            Sim2simCfg.robot_config.joint_names[name]
            for name in Sim2simCfg.robot_config.joint_names.keys()
        ],
        dtype=np.float32,
    )
    # gait command
    obs_gait_command = np.array(
        [
            Sim2simCfg.robot_config.gait_freq,
            Sim2simCfg.robot_config.gait_phase,
            Sim2simCfg.robot_config.gait_duration,
        ],
        dtype=np.float32,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # reset simulation
        mujoco.mj_resetData(model, data)

        # set initial joint positions and height
        for i, name in enumerate(Sim2simCfg.robot_config.joint_names):
            addr = model.joint(name).qposadr
            data.qpos[addr] = initial_joint_pos[i]
        data.qpos[2] = Sim2simCfg.robot_config.initial_height

        # forward pass and inital buffer fill
        mujoco.mj_forward(model, data)
        fall_status = False
        init_obs_list, fall_status = get_observation(
            model,
            data,
            np.zeros(6, dtype=np.float32),
            0.0,
            cmd_vel,
            initial_joint_pos,
            obs_gait_command,
        )
        for _ in range(Sim2simCfg.robot_config.obs_history_len):
            history_buffer.update_history(init_obs_list)

        # camera settings
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 10.0  # Zoom out
        viewer.cam.azimuth = 135  # Rotate camera (0 = Behind, 90 = Right Side).
        viewer.cam.elevation = -20  # Look slightly down

        while viewer.is_running():
            # Check if we are still in Warmup
            is_warmup = (time.time() - start_time) < warmup_delay

            # decimation loop
            if step_counter % Sim2simCfg.sim_config.decimation == 0:
                # update gait clock
                gait_time_accumulator += (
                    Sim2simCfg.sim_config.sim_dt * Sim2simCfg.sim_config.decimation
                )

                # get observation
                current_obs_list, fall_status = get_observation(
                    model,
                    data,
                    last_actions,
                    gait_time_accumulator,
                    cmd_vel,
                    initial_joint_pos,
                    obs_gait_command,
                )

                # update history buffer
                history_buffer.update_history(current_obs_list)
                stacked_obs = history_buffer.get_stacked_obs()

                # wait for warmup and then perform inference
                if is_warmup:
                    actions = np.zeros(6, dtype=np.float32)
                else:
                    obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).float()
                    with torch.no_grad():
                        actions = policy(obs_tensor)
                        actions = actions.detach().cpu().numpy().flatten()

                # update last actions
                actions = np.clip(actions, -100.0, 100.0)
                last_actions = actions

            # check if the robot has fallen
            # if fall_status:
            #     print("Robot has fallen down...Exiting")
            #     break

            # ---- physics step ----
            targets = (
                last_actions * Sim2simCfg.robot_config.action_scale
            ) + initial_joint_pos
            data.ctrl[:] = targets

            # step simulation
            mujoco.mj_step(model, data)

            # update viewer
            viewer.sync()

            # update step counter
            step_counter += 1

            time_until_next_step = model.opt.timestep - (time.time() - real_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            real_start_time = time.time()


def main():
    global cmd_vel, relative_policy_path

    # path variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    rl_model_path = os.path.join(project_root, relative_policy_path)
    robot_model_path = os.path.join(script_dir, "mujoco_xml", "SF_biped.xml")

    # run simulation
    run_mujoco(rl_model_path, robot_model_path, cmd_vel)


if __name__ == "__main__":
    main()
