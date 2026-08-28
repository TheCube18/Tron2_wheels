# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the WF-TRON2B wheeled biped.

The robot is a ~35.5 kg two-legged platform that rolls on a driven wheel at the end of
each leg. Every leg is a 5-DOF chain::

    base_Link
      -> proximal_pitch_?_Joint   (hip pitch, axis ~ +Y)
      -> proximal_roll_?_Joint    (hip roll,  axis +X)
      -> proximal_yaw_?_Joint     (hip yaw,   axis +Z)
      -> knee_?_Joint             (knee,      axis +Y)
      -> wheel_?_Joint            (wheel,     axis +Y, continuous)

so the articulation has 10 DOF in total: 8 positioned leg joints and 2 velocity-driven
wheels. All limits, effort/velocity caps and armatures below are taken straight from
``assets/WF_TRON2B/urdf/robot.urdf`` and ``assets/WF_TRON2B/xml/robot.xml``.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Asset paths.
##

TRON2_ASSET_DIR = os.path.join(os.path.dirname(__file__), "WF_TRON2B")
"""Absolute path to the WF-TRON2B asset folder (meshes / urdf / usd / xml)."""

TRON2_USD_PATH = os.path.join(TRON2_ASSET_DIR, "usd", "robot.usd")
"""Absolute path to the USD stage imported from the URDF."""

##
# Joint groups.
#
# The env resolves these with ``find_joints(..., preserve_order=True)``, so the order here
# is the order of the action vector. Keep leg joints first and wheels last.
##

TRON2_LEG_JOINT_NAMES = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
]
"""The 8 position-controlled leg joints, left leg then right leg."""

TRON2_WHEEL_JOINT_NAMES = ["wheel_L_Joint", "wheel_R_Joint"]
"""The 2 velocity-controlled wheel joints."""

##
# Nominal stance.
#
# knee = -2 * pitch keeps the shank vertical, so the wheel sits under the hip. Both legs take
# identical angles and 3D FK on the URDF puts both wheels at the same x to within 0.001 mm -
# there is no left/right stagger at any pose. What reads as a lunge is knee protrusion: at the
# old 30/-60 squat each knee juts 172 mm behind the hip line. At 10/-20 that drops to 57 mm
# while keeping the joint off its 0-rad singularity, where the knee has no leverage to push.
##

TRON2_NOMINAL_JOINT_POS = {
    "proximal_pitch_L_Joint": 0.1745,  # +10 deg
    "proximal_roll_L_Joint": 0.0,
    "proximal_yaw_L_Joint": 0.0,
    "knee_L_Joint": -0.3491,  # -20 deg
    "proximal_pitch_R_Joint": 0.1745,
    "proximal_roll_R_Joint": 0.0,
    "proximal_yaw_R_Joint": 0.0,
    "knee_R_Joint": -0.3491,
    "wheel_L_Joint": 0.0,
    "wheel_R_Joint": 0.0,
}
"""Default joint positions - upright with a slight knee bend, both wheels flat on the ground."""

TRON2_NOMINAL_BASE_HEIGHT = 0.7427
"""Height of the base origin above the ground in the nominal stance [m]."""

TRON2_NOMINAL_TRACK_WIDTH = 0.4233
"""Distance between the two wheel centres in the nominal stance [m].

Hip roll is heavily leveraged - about 1.35 m of track per radian - so this moves fast:
0.1 rad of symmetric roll already widens the stance to 0.559 m.
"""

##
# Articulation configuration.
##

TRON2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRON2_USD_PATH,
        # required for the ContactSensor the env uses to detect non-wheel ground contact
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, TRON2_NOMINAL_BASE_HEIGHT + 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=TRON2_NOMINAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    # keep resets away from the hard stops, where PhysX gets stiff
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip pitch / hip roll / knee - the heavy 200 Nm joints that carry the robot.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["proximal_pitch_._Joint", "proximal_roll_._Joint", "knee_._Joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=17.0693,
            stiffness=120.0,
            damping=5.0,
            armature=0.0754,
        ),
        # Hip yaw - a smaller 70 Nm joint, softer gains so it does not fight the others.
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["proximal_yaw_._Joint"],
            effort_limit_sim=70.0,
            velocity_limit_sim=16.3363,
            stiffness=60.0,
            damping=3.0,
            armature=0.0178,
        ),
        # Wheels - velocity controlled, so zero stiffness and damping acts as the vel gain.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_._Joint"],
            effort_limit_sim=22.0,
            velocity_limit_sim=41.89,
            stiffness=0.0,
            damping=2.0,
            armature=0.0110718,
        ),
    },
)
"""Configuration for the WF-TRON2B wheeled biped."""
