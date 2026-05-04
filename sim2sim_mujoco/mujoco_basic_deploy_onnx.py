"""Script for sim2sim deployment of a trained RL policy in Mujoco environment (ONNX Runtime)"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import onnxruntime as ort
from utils import HistoryBuffer, get_mujoco_data


# command velocity
cmd_vel = np.array([0.0, 0.0, 0.0])

relative_policy_path = (
    "logs/rsl_rl/bipedal_locomotion/2026-02-12_08-33-37_flat/exported/policy.onnx"
)


class Sim2simCfg:
    class sim_config:
        sim_dt = 1 / 200
        decimation = 4

    class robot_config:
        # observation
        obs_history_len = 5
        num_obs_terms = 8
        lin_vel_scale = 1.0
        ang_vel_scale = 1.0
        dof_pos_scale = 1.0
        dof_vel_scale = 1.0

        # joints
        joint_names = {
            "left_hip_pitch_joint": 0.3,
            "right_hip_pitch_joint": -0.3,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_knee_joint": 0.6,
            "right_knee_joint": -0.6,
        }
        initial_height = 0.53
        action_scale = 0.25

        # gait
        gait_freq = 1.75  # [Hz]
        gait_phase = 0.5  # [0-1]
        gait_duration = 0.5  # [0-1]

        # mujoco model gains
        stiffness_gain = 40.0
        damping_gain = -2.5

    class runtime_config:
        # Host: CUDA with CPU fallback. Swap to TensorRT on Jetson.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # providers = [
        #     ("TensorrtExecutionProvider", {
        #         "trt_fp16_enable": True,
        #         "trt_engine_cache_enable": True,
        #         "trt_engine_cache_path": "./trt_cache",
        #     }),
        #     "CUDAExecutionProvider",
        #     "CPUExecutionProvider",
        # ]


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
        obs_ang_vel.reshape(-1),
        obs_proj_gravity.reshape(-1),
        obs_cmd.reshape(-1),
        obs_joint_pos.reshape(-1),
        obs_joint_vel.reshape(-1),
        obs_last_actions.reshape(-1),
        obs_gait_phase_sin_cos.reshape(-1),
        obs_gait_command.reshape(-1),
    ]

    fall_status = False
    if abs(obs_proj_gravity[0]) > 0.90:
        fall_status = True

    return current_obs, fall_status


def load_policy(rl_model_path):
    """Load ONNX policy and return session + input/output names."""
    print(f"Loading ONNX Policy: {rl_model_path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        rl_model_path,
        sess_options=sess_options,
        providers=Sim2simCfg.runtime_config.providers,
    )

    print(f"Active providers: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape} | Output: {output_name}")

    return session, input_name, output_name


def run_mujoco(rl_model_path, robot_model_path, cmd_vel):
    # load policy
    session, input_name, output_name = load_policy(rl_model_path)

    # load model
    print(f"Loading Model: {robot_model_path}")
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)

    # set gains
    model.actuator_gainprm[:, 0] = Sim2simCfg.robot_config.stiffness_gain
    model.actuator_biasprm[:, 2] = Sim2simCfg.robot_config.damping_gain

    # init history buffer
    history_buffer = HistoryBuffer(
        obs_history_len=Sim2simCfg.robot_config.obs_history_len,
        num_obs_terms=Sim2simCfg.robot_config.num_obs_terms,
    )

    # loop variables
    step_counter = 0
    last_actions = np.zeros(6, dtype=np.float32)
    gait_time_accumulator = 0.0

    # time related variables
    start_time = time.time()
    real_start_time = time.time()
    warmup_delay = 0.10

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

        # forward pass and initial buffer fill
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

        # warm up ORT session (first call allocates GPU mem / builds TRT engine)
        dummy_obs = history_buffer.get_stacked_obs()
        dummy_input = np.expand_dims(dummy_obs, axis=0).astype(np.float32)
        _ = session.run([output_name], {input_name: dummy_input})
        print("ONNX session warmed up.")

        # camera settings
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 10.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
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
                    # ONNX inference: pure numpy, no torch
                    obs_input = np.expand_dims(stacked_obs, axis=0).astype(np.float32)
                    outputs = session.run([output_name], {input_name: obs_input})
                    actions = outputs[0].flatten().astype(np.float32)

                # update last actions
                actions = np.clip(actions, -100.0, 100.0)
                last_actions = actions

            # check if the robot has fallen
            if fall_status:
                print("Robot has fallen down...Exiting")
                break

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