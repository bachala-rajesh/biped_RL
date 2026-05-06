class Sim2simCfg:
    class sim_config:
        # Policy control period — MUST equal training-time `sim.dt * decimation`
        # (Isaac: 1/200 * 4 = 0.020 s, i.e. policy at 50 Hz).
        # The deploy script derives MuJoCo decimation from this and the XML timestep.
        policy_dt = 0.020

        # Legacy fields, kept for backward compat with mujoco_basic_deploy_v2.py
        # and mujoco_keyb_teleport.py. New code should use policy_dt instead.
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

        # joints — order MUST match training action order exactly
        joint_names = {
            "left_hip_pitch_joint": 0.3,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.6,
            "right_hip_pitch_joint": -0.3,
            "right_hip_roll_joint": 0.0,
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
        damping_gain = -4.0
        kp = 45e.0
        kd = 4.0
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
