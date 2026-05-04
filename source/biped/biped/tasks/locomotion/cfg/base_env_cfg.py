# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise


from biped.tasks.locomotion import mdp
from biped.tasks.locomotion.cfg.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
)
from biped.assets.config.simple_biped_config import BIPED_CONFIG

##
# Scene definition
##

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = MISSING

    height_scanner: RayCasterCfg = MISSING

    # robot
    # robot = BIPED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            # texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##
# mdp components
##

@configclass
class CommandCfg:
    """Command terms for the MDP"""

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        debug_vis=True,
        heading_control_stiffness=0.5,
        resampling_time_range=(0.0, 5.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
        ],
        scale=0.25,
        use_default_offset=True,
        preserve_order= True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        # base_lin_vel = ObsTerm(func=mdp.base_lib_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=UniformNoise(operation="add", n_min=-0.2, n_max=0.2),
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05),
        )

        # velocity command
        vel_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=UniformNoise(operation="add", n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=UniformNoise(operation="add", n_min=-0.05, n_max=0.05),
        )

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # gaits
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(
            func=mdp.get_gait_command, params={"command_name": "gait_command"}
        )

        # heights scan
        heights: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True         # TODO: change it to False for stackinf frame by frame observations

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # Policy observation

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)

        vel_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(
            func=mdp.get_gait_command, params={"command_name": "gait_command"}
        )

        heights: ObsTerm = MISSING

        # Privileged observation
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*knee_link"
                ),
            },
        )

        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_propertirs = ObsTerm(func=mdp.robot_material_properties)
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="(left|right).*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),
            "damping_distribution_params": (3.0, 5.0),
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.0, 0.0),
                "z": (-0.5, 0.5),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # interval
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-100.0, 100.0),
                "y": (-100.0, 100.0),
                "z": (-0.0, 0.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task Rewards --
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # rew_no_fly = RewTerm(
    #     func=mdp.no_fly,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_link"),
    #         "threshold": 5.0,
    #     },
    # )

    # Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -1.0,
            "tracking_contacts_shaped_vel": -1.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*knee_link"),
        },
    )

    # penalizations
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*hip_pitch_link", ".*hip_roll_link", "base_link"],
            ),
            "threshold": 10.0,
        },
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-2.0e-05)
    pen_base_height = RewTerm(
        func=mdp.base_com_height,
        params={
            "target_height": 0.49,
        },
        weight=-1.0,
    )
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-05)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class BipedEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandCfg = CommandCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
