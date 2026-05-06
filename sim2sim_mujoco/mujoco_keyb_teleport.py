# import math
import numpy as np
import mujoco
import mujoco.viewer
import mujoco_viewer
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
from utils import HistoryBuffer, get_mujoco_data
from keyboard_cmd import cmd, start_keyboard_listener
from sim_config import Sim2simCfg

relative_policy_path = (
    "logs/rsl_rl/bipedal_locomotion/2026-05-04_13-44-30_flat/exported/policy.pt"
)




# --- Data collection lists for plotting (CONTROL ONLY) ---
time_data = []
commanded_joint_pos_data = []
actual_joint_pos_data = []
actions_data = []
commanded_lin_vel_x_data = []
commanded_lin_vel_y_data = []
commanded_ang_vel_z_data = []
actual_lin_vel_data = []
actual_ang_vel_data = []


def get_observation(
    model, data, last_actions, gait_time, cmd_vel, initial_joint_pos, obs_gait_command
):
    proj_gravity, lin_vel, ang_vel, joints_pos, joints_vel = get_mujoco_data(
        model, data, Sim2simCfg.robot_config.joint_names
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
    obs_proj_gravity = proj_gravity

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

    return current_obs, lin_vel, ang_vel, fall_status


def run_mujoco(rl_policy_path, robot_model_path):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    # Start keyboard listener
    keyboard_listener = start_keyboard_listener()

    # load policy model
    policy = torch.jit.load(rl_policy_path)
    policy.eval()

    # load mujoco model
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

        # save initial qpos and qvel
        initial_qpos = data.qpos.copy()
        initial_qvel = data.qvel.copy()

        # set initial joint positions and height
        for i, name in enumerate(Sim2simCfg.robot_config.joint_names):
            addr = model.joint(name).qposadr
            data.qpos[addr] = initial_joint_pos[i]
        data.qpos[2] = Sim2simCfg.robot_config.initial_height

        # forward pass and inital buffer fill
        cmd_vel = np.array([cmd.vx, cmd.vy, cmd.dyaw], dtype=np.float32)
        mujoco.mj_forward(model, data)
        fall_status = False
        init_obs_list, lin_vel, ang_vel, fall_status = get_observation(
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
        viewer.cam.distance = 5.0  # Zoom out
        viewer.cam.azimuth = 135  # Rotate camera (0 = Behind, 90 = Right Side).
        viewer.cam.elevation = -20  # Look slightly down

        #  time related variables
        start_time = time.time()
        real_start_time = time.time()
        warmup_delay = 0.10  # Wait few seconds before turning on Policy

        while viewer.is_running():
            # Check if we are still in Warmup
            is_warmup = (time.time() - start_time) < warmup_delay

            if cmd.reset_requested:
                print("Performing reset: restoring qpos/qvel and zeroing commands")
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                # clear commands and history
                cmd.reset()
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)
                cmd.reset_requested = False

            # decimation loop
            if step_counter % Sim2simCfg.sim_config.decimation == 0:
                # update gait clock
                gait_time_accumulator += (
                    Sim2simCfg.sim_config.sim_dt * Sim2simCfg.sim_config.decimation
                )

                # get observation
                cmd_vel = np.array([cmd.vx, cmd.vy, cmd.dyaw], dtype=np.float32)
                current_obs_list, lin_vel, ang_vel, fall_status = get_observation(
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

                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [
                        float(base_pos[0]),
                        float(base_pos[1]),
                        float(base_pos[2]),
                    ]

            # ---- physics step ----
            targets = (
                last_actions * Sim2simCfg.robot_config.action_scale
            ) + initial_joint_pos
            data.ctrl[:] = targets

            # step simulation
            mujoco.mj_step(model, data)

            # --- Collect low-frequency data for plotting (CONTROL ONLY) ---
            time_data.append(gait_time_accumulator)
            commanded_joint_pos_data.append(targets.copy())
            current_actual_joints = []
            for name in Sim2simCfg.robot_config.joint_names:
                addr = model.joint(name).qposadr
                current_actual_joints.append(data.qpos[addr])

            actual_joint_pos_data.append(np.array(current_actual_joints))
            actions_data.append(last_actions.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(
                lin_vel[:2].copy()
            )  # x and y linear velocity in the base frame
            actual_ang_vel_data.append(ang_vel[2].copy())  # z angular velocity

            # update viewer
            viewer.sync()

            # update step counter
            step_counter += 1

            time_until_next_step = model.opt.timestep - (time.time() - real_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            real_start_time = time.time()

    viewer.close()

    # Stop keyboard listener
    keyboard_listener.stop()


def plot_data(logs_dir, rl_policy_path):
    global \
        time_data, \
        commanded_joint_pos_data, \
        actual_joint_pos_data, \
        actions_data, \
        commanded_lin_vel_x_data, \
        commanded_lin_vel_y_data, \
        commanded_ang_vel_z_data, \
        actual_lin_vel_data, \
        actual_ang_vel_data

    # Setup save paths: logs dir in sim2sim_mujoco, filename with Kp, Kd and policy name
    rl_policy_path = Path(rl_policy_path)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = f"{rl_policy_path.parent.parent.name}_Kp{Sim2simCfg.robot_config.stiffness_gain}_Kd{Sim2simCfg.robot_config.damping_gain}"

    # Convert collected data to numpy arrays
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    actions_data = np.array(actions_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)

    # Plot 1: Commanded vs Actual Joint Positions
    num_joints = len(Sim2simCfg.robot_config.joint_names.keys())
    n_cols = 3  # Or adjust based on num_joints
    n_rows = (num_joints + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    joint_names = Sim2simCfg.robot_config.joint_names.keys()

    for i, name in enumerate(joint_names):
        ax = axes1[i]
        ax.plot(
            time_data,
            commanded_joint_pos_data[:, i],
            label="Commanded",
            linestyle="--",
            color="red",
        )
        ax.plot(time_data, actual_joint_pos_data[:, i], label="Actual", color="blue")
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")

    # global title
    fig1.suptitle(
        f"Joint Tracking Performance (Kp={Sim2simCfg.robot_config.stiffness_gain}, Kd={Sim2simCfg.robot_config.damping_gain})",
        fontsize=16,
    )
    plt.tight_layout()
    fig1_path = logs_dir / f"{save_prefix}_joint_positions.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"Saved: {fig1_path}")

    # PLOT 2: Commanded vs Actual Base Velocities
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Linear Velocity X
    axes2 = axes2.flatten()
    axes2[0].plot(
        time_data,
        commanded_lin_vel_x_data,
        label="Commanded Vx",
        linestyle="--",
        color="red",
    )
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label="Actual Vx", color="blue")
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    # Linear Velocity Y
    axes2[1].plot(
        time_data,
        commanded_lin_vel_y_data,
        label="Commanded Vy",
        linestyle="--",
        color="red",
    )
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label="Actual Vy", color="blue")
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    # Angular Velocity Z
    axes2[2].plot(
        time_data,
        commanded_ang_vel_z_data,
        label="Commanded Dyaw",
        linestyle="--",
        color="red",
    )
    axes2[2].plot(time_data, actual_ang_vel_data, label="Actual Dyaw", color="blue")
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    # global title
    fig2.suptitle(
        f"Base Velocity Tracking Performance (Kp={Sim2simCfg.robot_config.stiffness_gain}, Kd={Sim2simCfg.robot_config.damping_gain})",
        fontsize=16,
    )
    plt.tight_layout()
    fig2_path = logs_dir / f"{save_prefix}_base_velocities.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"Saved: {fig2_path}")

    # view plots
    plt.show()


def main():
    global relative_policy_path

    # path variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    rl_policy_path = os.path.join(project_root, relative_policy_path)
    robot_model_path = os.path.join(script_dir, "mujoco_xml", "SF_biped.xml")

    # run simulation
    print(f"Running Simulation with Policy: {rl_policy_path}")
    print(f"Running Simulation with Model: {robot_model_path}")
    run_mujoco(rl_policy_path, robot_model_path)

    #  Plotting Section
    print("Simulation finished. Generating plots...")

    logs_dir_path = Path(script_dir) / "log_plots"
    plot_data(logs_dir_path, rl_policy_path)


if __name__ == "__main__":
    main()
