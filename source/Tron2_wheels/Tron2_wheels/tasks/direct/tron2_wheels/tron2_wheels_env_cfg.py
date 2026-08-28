# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from Tron2_wheels.assets.tron2 import (
    TRON2_CFG,
    TRON2_LEG_JOINT_NAMES,
    TRON2_NOMINAL_BASE_HEIGHT,
    TRON2_NOMINAL_TRACK_WIDTH,
    TRON2_WHEEL_JOINT_NAMES,
)


@configclass
class EventCfg:
    floor_material = EventTermCfg(
    func=mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="wheel.*"),
        "static_friction_range": (0.65, 1.0),
        "dynamic_friction_range": (0.5, 0.85),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
        "make_consistent": True,
    },


    # push_robot = EventTermCfg(
    #     func = mdp.push_by_setting_velocity,
    #     mode = "interval",
    #     interval_range_s = (2, 60), # 2~60 sec
    #     params={"velocity_range": {"x": (-1, 1), "y": (-1, 1)}},
    # )
)



@configclass
class Tron2WheelsEnvCfg(DirectRLEnvCfg):
    """Velocity-tracking task for the WF-TRON2B wheeled biped."""

    # env
    decimation = 4  # 200 Hz physics / 4 -> 50 Hz policy
    episode_length_s = 20.0

    # - spaces definition
    #   action  = 8 leg position offsets + 2 wheel velocity targets
    #   observation:
    #       base angular velocity   (3)
    #       projected gravity       (3)
    #       velocity command        (3)
    #       leg joint pos - default (8)   wheel angles are meaningless, so they are excluded
    #       joint velocities        (10)
    #       last action             (10)
    action_space = 10
    observation_space = 37
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = TRON2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensor(s) - used to detect anything but a wheel touching the ground
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # ground - wheels need grip, so override the default 0.5 friction
    ground_cfg: GroundPlaneCfg = GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    events: EventCfg = EventCfg()

    leg_joint_names = TRON2_LEG_JOINT_NAMES
    wheel_joint_names = TRON2_WHEEL_JOINT_NAMES

    undesired_contact_body_names = ["base_Link", "proximal_.*_Link", "knee_.*_Link"]

    action_scale_leg = 0.5  # [rad] offset applied on top of the nominal stance
    action_scale_wheel = 10.0  # [rad/s] wheel velocity target (~1 m/s per rad/s * 0.1 m radius)
    action_clip = 100.0  # safety clip on the raw policy output, before scaling

    obs_scale_ang_vel = 0.25
    obs_scale_joint_pos = 1.0
    obs_scale_joint_vel = 0.05

    # - velocity command [lin_x, lin_y, ang_z], resampled on every reset.
    #   forward speed plus yaw rate is full planar mobility for a differential drive, so this
    #   covers every direction the robot can actually go. lin_y stays 0: the wheels cannot
    #   slide sideways, so commanding it would only ask for something unreachable.
    lin_vel_x_range = (-1.0, 1.0)  # [m/s]
    lin_vel_y_range = (0.0, 0.0)  # [m/s]
    ang_vel_z_range = (-1.0, 1.0)  # [rad/s]

    rew_scale_ang_vel = -0.05

    rew_scale_alive = 0.5
    rew_scale_upright = -1.0

    rew_scale_action = -0.01
    rew_scale_action_rate = -0.01

    # holds the 8 leg joints near the default stance. max deviation is 0.5 rad/joint
    # (action_scale_leg), so the worst case sum is 8 * 0.5^2 = 2.0 -> -1.0 reward
    rew_scale_joint_deviation = -0.5


    rew_scale_track_lin = 1.8
    rew_scale_track_ang = 0.5

    tracking_sigma_lin = 0.14 # cant be change too strict it will break the control
    tracking_sigma_ang = 0.25

    
    min_base_height = 0.35  # [m] the base collapsed if it drops below this
    min_up_projection = 0.5  # tipped over past ~60 deg from upright
    contact_force_threshold = 1.0  # [N] net contact force that counts as a real touch


    nominal_base_height = TRON2_NOMINAL_BASE_HEIGHT  # [m]
    rew_scale_height = -0.5

    # stance width - points while the two wheels stay within tolerance of the nominal
    # track, nothing outside it. roll gives ~1.35 m of track per rad, so +-0.10 m is
    # about +-4.2 deg of symmetric roll before the bonus cuts out.
    wheel_body_names = ["wheel_L_Link", "wheel_R_Link"]
    stance_width_target = TRON2_NOMINAL_TRACK_WIDTH  # [m]
    stance_width_tolerance = 0.10  # [m]
    rew_scale_stance_width = 0.5


    rew_scale_pose_match = -1.4 # 1.2



    debug_arrow = True  # cyan = actual heading, red = commanded; GUI only

    # keyboard teleop - drives the command by hand instead of resampling it.
    # off by default so training is untouched; turn on for play with env.use_keyboard=true
    use_keyboard = False
    keyboard_v_x_sensitivity = 1.0  # [m/s] per key press
    keyboard_v_y_sensitivity = 0.0  # no strafe: the arrow keys are rebound to yaw instead
    keyboard_omega_z_sensitivity = 1.0  # [rad/s] per key press
