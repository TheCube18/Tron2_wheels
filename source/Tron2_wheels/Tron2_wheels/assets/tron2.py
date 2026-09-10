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

import numpy as np

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

##
# Height kinematics.
#
# The stance is solved so the whole-robot COM sits directly over the wheel contact at every
# commanded height. knee = -2 * pitch does NOT do this: it keeps the *wheel* under the *hip*,
# but 13.4 kg of base rides 0.111 m above the base origin, so raking the thigh back carries
# the mass behind the contact patch - 83 mm of lean-back at a 0.35 m stance, and only ~2 mm
# at full extension. That mismatch makes the bot statically stable when crouched and neutral
# when straight, which is why it crept forward standing tall and sat still folded.
#
# Solving instead for (pitch, knee) that satisfies both the target height and zero COM offset
# is exactly determined - two joints, two constraints - and the solution is unique, monotonic
# in pitch, and stays inside the joint limits over the whole range.
##

_LEG_CHAIN = (
    # (offset from parent, joint axis) down the left leg, straight from the URDF
    ((-0.00044, 0.09497, 0.05446), (0.0, 0.9961947268929313, -0.0871554135479712)),
    ((-0.0352, 0.07261, -0.00635), (1.0, 0.0, 0.0)),
    ((0.0352, 0.0, -0.1348), (0.0, 0.0, 1.0)),
    ((0.004, 0.0319, -0.2152), (0.0, 1.0, 0.0)),
    ((-0.004, 0.0131, -0.35), (0.0, 1.0, 0.0)),
)

TRON2_WHEEL_RADIUS = 0.1
"""Wheel collision radius [m]."""

# mass and COM of each leg link, in chain order, from the URDF inertials
_LINK_MASS = (2.05, 1.75, 1.6, 4.72, 0.95)
_LINK_COM = (
    (0.0033, 0.06714, -0.00603),
    (0.03623, 0.00056, -0.08095),
    (-0.00241, 0.00149, -0.10876),
    (-0.00233, -0.02777, -0.12898),
    (0.00104, 0.00488, 0.00083),
)

# base_Link plus the imu. this is 13.4 of the 35.55 kg total, and it sits high, which is why
# it dominates where the COM ends up
_BASE_MASS = 13.41
_BASE_COM = (0.00241, -0.00022, 0.11116)

_KNEE_LIMITS = (-2.6179938779914944, 0.2617993877991494)


def _axis_rot(axis, angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def tron2_leg_state(pitch: float, knee: float) -> tuple[float, float]:
    """COM offset ahead of the wheel contact, and base height, for a symmetric stance."""
    angles = (pitch, 0.0, 0.0, knee, 0.0)
    pos, rot = np.zeros(3), np.eye(3)
    total = _BASE_MASS
    moment = _BASE_MASS * np.asarray(_BASE_COM, dtype=float)
    for i, ((offset, axis), angle) in enumerate(zip(_LEG_CHAIN, angles)):
        pos = pos + rot @ np.asarray(offset, dtype=float)
        rot = rot @ _axis_rot(axis, angle)
        # the right leg mirrors in y, so it contributes identically in x and z
        moment = moment + 2.0 * _LINK_MASS[i] * (pos + rot @ np.asarray(_LINK_COM[i], dtype=float))
        total += 2.0 * _LINK_MASS[i]
    com = moment / total
    return float(com[0] - pos[0]), float(-pos[2] + TRON2_WHEEL_RADIUS)


def _knee_balancing_com(pitch: float, iters: int = 60) -> float:
    """Bisect for the knee angle that puts the COM directly over the wheel contact."""
    lo, hi = _KNEE_LIMITS
    f_lo = tron2_leg_state(pitch, lo)[0]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (tron2_leg_state(pitch, mid)[0] < 0.0) == (f_lo < 0.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tron2_balanced_stance_table(num_samples: int = 128, max_pitch: float = 1.55):
    """(pitch, knee, height) with the COM over the wheel, sorted by ascending height."""
    pitch = np.linspace(0.0, max_pitch, num_samples)
    knee = np.array([_knee_balancing_com(float(p)) for p in pitch])
    height = np.array([tron2_leg_state(float(p), float(k))[1] for p, k in zip(pitch, knee)])
    order = np.argsort(height)
    return pitch[order], knee[order], height[order]


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
