# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_yaw, quat_from_angle_axis, sample_uniform

from .tron2_wheels_env_cfg import Tron2WheelsEnvCfg


class Tron2WheelsEnv(DirectRLEnv):
    """Velocity-tracking task for the WF-TRON2B wheeled biped.

    The policy outputs position offsets for the 8 leg joints and velocity targets for the
    2 wheels, and is asked to track a randomly sampled base velocity command while staying
    upright on its wheels.
    """

    cfg: Tron2WheelsEnvCfg

    def __init__(self, cfg: Tron2WheelsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve the joint groups once. ``preserve_order`` keeps them aligned with the name
        # lists in the cfg, so the action layout is exactly [8 leg joints, 2 wheels].
        self._leg_dof_idx, _ = self.robot.find_joints(self.cfg.leg_joint_names, preserve_order=True)
        self._wheel_dof_idx, _ = self.robot.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        self._all_dof_idx = self._leg_dof_idx + self._wheel_dof_idx
        self._num_leg_dof = len(self._leg_dof_idx)
        self._num_wheel_dof = len(self._wheel_dof_idx)
        self._num_actions = self._num_leg_dof + self._num_wheel_dof

        # Bodies whose ground contact means the robot fell over.
        self._undesired_contact_body_idx, _ = self.contact_sensor.find_bodies(self.cfg.undesired_contact_body_names)

        # The two wheels, for measuring how far apart the feet are.
        self._wheel_body_idx, _ = self.robot.find_bodies(self.cfg.wheel_body_names, preserve_order=True)

        # Action buffers.
        self.actions = torch.zeros(self.num_envs, self._num_actions, device=self.device)
        self._previous_actions = torch.zeros_like(self.actions)

        #commands - filled per env by _resample_commands on every reset
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # keyboard teleop. when active it overwrites the command every step, so the random
        # resampling on reset is skipped and every env follows the same typed command.
        self._keyboard = None
        if self.cfg.use_keyboard and self.sim.has_gui():
            self._keyboard = Se2Keyboard(
                Se2KeyboardCfg(
                    sim_device=self.device,
                    v_x_sensitivity=self.cfg.keyboard_v_x_sensitivity,
                    v_y_sensitivity=self.cfg.keyboard_v_y_sensitivity,
                    omega_z_sensitivity=self.cfg.keyboard_omega_z_sensitivity,
                )
            )
            # Se2Keyboard binds left/right to a lateral strafe, which wheels cannot do.
            # Rebind them to yaw so the arrow keys alone are the whole control set.
            yaw = self.cfg.keyboard_omega_z_sensitivity
            for key, sign in (("LEFT", 1.0), ("NUMPAD_4", 1.0), ("RIGHT", -1.0), ("NUMPAD_6", -1.0)):
                self._keyboard._INPUT_KEY_MAPPING[key] = np.asarray([0.0, 0.0, sign * yaw])
            print(
                "\nTron2 keyboard teleop"
                f"\n  Up / Down    : drive forward / back  (+-{self.cfg.keyboard_v_x_sensitivity} m/s)"
                f"\n  Left / Right : turn                  (+-{yaw} rad/s)"
                "\n  Z / X        : turn, same as Left / Right"
                "\n  L            : zero the command\n"
            )

        self._leg_pos_targets = torch.zeros(self.num_envs, self._num_leg_dof, device=self.device)
        self._wheel_vel_targets = torch.zeros(self.num_envs, self._num_wheel_dof, device=self.device)


        # The nominal stance the leg actions are offsets from.
        self._default_leg_pos = self.robot.data.default_joint_pos[:, self._leg_dof_idx].clone()

        self._markers = None
        if self.cfg.debug_arrow and self.sim.has_gui():
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/tron2_markers",
                markers={
                    # 0 - direction the robot is actually travelling, CYAN
                    "actual": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.15, 0.15, 0.3),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                    ),
                    # 1 - direction it was commanded to travel, RED
                    "target": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                        scale=(0.15, 0.15, 0.3),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)
            # 0.35 above the base floats the arrows clear of the torso
            self._marker_offset = torch.tensor([0.0, 0.0, 0.35], device=self.device)
            self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground_cfg)
        self.scene.clone_environments(copy_from_source=False)

       
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._keyboard is not None:
            self._commands[:] = self._keyboard.advance()

        self._previous_actions = self.actions
        self.actions = actions.clone().clamp(-self.cfg.action_clip, self.cfg.action_clip)

        
        self._leg_pos_targets = (
            self._default_leg_pos + self.cfg.action_scale_leg * self.actions[:, : self._num_leg_dof]
        )
        self._wheel_vel_targets = self.cfg.action_scale_wheel * self.actions[:, self._num_leg_dof :]

        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._leg_pos_targets, joint_ids=self._leg_dof_idx)
        self.robot.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._wheel_dof_idx)


    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot.data.root_ang_vel_b * self.cfg.obs_scale_ang_vel, 
                self.robot.data.projected_gravity_b, 
                self._commands,  
                # joints - wheel angles spin without bound, so only the legs report position
                (self.robot.data.joint_pos[:, self._leg_dof_idx] - self._default_leg_pos)
                * self.cfg.obs_scale_joint_pos,  


                self.robot.data.joint_vel[:, self._all_dof_idx] * self.cfg.obs_scale_joint_vel,  
                self.actions, 
            ),
            dim=-1,
        )
        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:

        total_reward = compute_rewards(
            self.robot.data.projected_gravity_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.root_lin_vel_b,

            self.actions,
            self._previous_actions,

            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_ang_vel,

            self.cfg.rew_scale_action,
            self.cfg.rew_scale_action_rate,

            self.cfg.rew_scale_alive,

            self._commands,

            self.cfg.rew_scale_track_lin,
            self.cfg.rew_scale_track_ang,
            self.cfg.tracking_sigma_lin,
            self.cfg.tracking_sigma_ang,

            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2],
            self.cfg.nominal_base_height,
            self.cfg.rew_scale_height,

            self._stance_width(),
            self.cfg.stance_width_target,
            self.cfg.stance_width_tolerance,
            self.cfg.rew_scale_stance_width,

            self.robot.data.joint_pos[:, self._leg_dof_idx] - self._default_leg_pos,
            self.cfg.rew_scale_pose_match,
            )

        return total_reward

    def _stance_width(self) -> torch.Tensor:
        # straight-line distance between the wheel centres. taken in world frame on purpose:
        # a distance is rotation invariant, so this reads the same however the base is leaning.
        wheel_pos = self.robot.data.body_pos_w[:, self._wheel_body_idx]
        return torch.norm(wheel_pos[:, 0] - wheel_pos[:, 1], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        #    tipped over
        tipped = self.robot.data.projected_gravity_b[:, 2] > -self.cfg.min_up_projection

        #    collapsed
        base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        too_low = base_height < self.cfg.min_base_height

        #    crashed. something other than a wheel is pushing against the ground.
        net_forces = self.contact_sensor.data.net_forces_w_history
        contact_forces = torch.norm(net_forces[:, :, self._undesired_contact_body_idx], dim=-1)
        undesired_contact = torch.any(
            torch.max(contact_forces, dim=1)[0] > self.cfg.contact_force_threshold, dim=1
        )

        died = tipped | too_low | undesired_contact
        return died, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if self.cfg.events:
            self.event_manager.reset(env_ids)

        
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        if self._keyboard is None:
            self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: Sequence[int]) -> None:
        """Draw a fresh velocity command for the given envs."""
        num_resets = len(env_ids)
        self._commands[env_ids, 0] = sample_uniform(*self.cfg.lin_vel_x_range, (num_resets,), self.device)
        self._commands[env_ids, 1] = sample_uniform(*self.cfg.lin_vel_y_range, (num_resets,), self.device)
        self._commands[env_ids, 2] = sample_uniform(*self.cfg.ang_vel_z_range, (num_resets,), self.device)

    def _visualize_markers(self) -> None:
        if self._markers is None:
            return

        actual_dir = self.robot.data.root_lin_vel_w[:, :2]

        # command is body-frame, so rotate by base yaw only - the full quaternion would
        # tilt the arrow as the robot pitches
        cmd_b = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_b[:, :2] = self._commands[:, :2]
        target_dir = quat_apply_yaw(self.robot.data.root_quat_w, cmd_b)[:, :2]

        actual_quat, actual_scale = self._arrow_from_xy(actual_dir)
        target_quat, target_scale = self._arrow_from_xy(target_dir)

        loc = self.robot.data.root_pos_w + self._marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((actual_quat, target_quat))
        scales = torch.vstack((actual_scale, target_scale))

        all_envs = torch.arange(self.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self._markers.visualize(loc, rots, scales, marker_indices=indices)

    def _arrow_from_xy(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # length scales with magnitude: atan2(0, 0) is 0, so a stationary robot would
        # otherwise show an arrow confidently pointing at world +x
        yaws = torch.atan2(xy[:, 1], xy[:, 0])
        quat = quat_from_angle_axis(yaws, self._up_dir)

        scale = torch.ones(xy.shape[0], 3, device=self.device)
        scale[:, 0] = (torch.linalg.norm(xy, dim=1) * 3.0).clamp(min=0.01, max=2.0)
        return quat, scale


#------ EDIT ----- :D
def upright_penalty(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    return projected_gravity_b[:, 0] ** 2 + projected_gravity_b[:, 1] ** 2


def height_penalty(root_height: torch.Tensor, target_height: float) -> torch.Tensor:
    return (root_height - target_height) ** 2


def pose_match_reward(leg_pos_offset: torch.Tensor) -> torch.Tensor:
    # leg_pos_offset is joint_pos - default, so the nominal stance is exactly 0 error.
    # norm over the 8 leg joints is a shared budget: one joint may swing the full tolerance,
    # or all eight may drift by tolerance/sqrt(8) each. wheels excluded - they spin unbounded. 
    return (torch.norm(leg_pos_offset, dim=1)).float()


def stance_width_reward(stance_width: torch.Tensor, target: float, tolerance: float) -> torch.Tensor:
    # 1.0 while the feet are within tolerance of the nominal track, 0.0 outside. no slope
    # either side, so there is nothing to gain by drifting toward the edge of the band.
    return (torch.abs(stance_width - target) <= tolerance).float()


def angular_velocity_penalty(ang_vel_b: torch.Tensor) -> torch.Tensor:
    # roll rate + pitch rate. yaw (index 2) stays out: that one is the command's business
    return ang_vel_b[:, 0] ** 2 + ang_vel_b[:, 1] ** 2

def action_penalty(actions: torch.Tensor) -> torch.Tensor:
    return torch.sum(actions ** 2, dim=1)

def action_rate_penalty(actions: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
    return torch.sum((actions - previous_actions) ** 2, dim=1)
def alive_bonus(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.ones(num_envs, device=device)

def track_lin_vel_xy(lin_vel_b: torch.Tensor, cmd: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian on the planar velocity error: exp(-|v_xy - cmd_xy|^2 / sigma).

    cmd is [v_x, v_y, omega_z]. v_y is a real commanded quantity now, not a hardcoded zero:
    with joint_deviation_penalty off the legs are free, and HAA (axis 1 0 0) swings them
    sideways with about 13.6 cm of reach at leg_action_scale 0.5, so lateral motion is
    something the robot can actually produce and therefore something worth commanding.

    sigma must be sized against the command range, not shared between the two terms. A typical
    |cmd| of 0.1 m/s wants sigma ~ 0.02 so that standing still scores about exp(-1) and there is
    a real slope to climb; at the old shared 0.03 it already scored 0.72 and the policy had
    almost nothing to gain by moving.
    """
    # squaring already discards sign, so an abs() would be redundant
    err = torch.sum((cmd[:, :2] - lin_vel_b[:, :2]) ** 2, dim=1)
    return torch.exp(-err / sigma)


def track_ang_vel_z(ang_vel_b: torch.Tensor, cmd: torch.Tensor, sigma: float) -> torch.Tensor:
    # typical |cmd| of 0.5 rad/s wants sigma ~ 0.25 for the same reason, in the other direction.
    # cmd index 2, not 1 -- the command is [v_x, v_y, omega_z] now that v_y is commanded
    err = (ang_vel_b[:, 2] - cmd[:, 2]) ** 2
    return torch.exp(-err / sigma)



#COMPUTE THE TOTAL REWARD

def compute_rewards(
    projected_gravity_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    lin_vel_b: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    rew_scale_upright: float,
    rew_scale_ang_vel: float,
    rew_scale_action: float,
    rew_scale_action_rate: float,
    rew_scale_alive: float,

    commands: torch.Tensor,
    rew_scale_track_lin: float,
    rew_scale_track_ang: float,
    tracking_sigma_lin: float,
    tracking_sigma_ang: float,

    root_height: torch.Tensor,
    target_height: float,
    rew_scale_height: float,

    stance_width: torch.Tensor,
    stance_width_target: float,
    stance_width_tolerance: float,
    rew_scale_stance_width: float,

    leg_pos_offset: torch.Tensor,
    rew_scale_pose_match: float,

) -> torch.Tensor:
    
    total = (
        #Deducted points
        rew_scale_upright * upright_penalty(projected_gravity_b) # bot keeping itself upright, the closer how straight it can get. Formula: (gx)² + (gy)² = sin²(bot_tilt)
        
        + rew_scale_ang_vel * angular_velocity_penalty(ang_vel_b) # evaluate the roll and pitch rate and penalize rapid rotation around the horizontal (X and Y) axes. Formula: ωx² + ωy² 
        + rew_scale_action * action_penalty(actions) # normalization for action, and penalize large action command, like slamming actuators, etc.. Formula : (current_action)²
        + rew_scale_action_rate * action_rate_penalty(actions, previous_actions)# compare previous action with current action, penalize action if there is rapid change between the two actions. Formula: (current_action - previous_action)²

        + rew_scale_height * height_penalty(root_height, target_height)# compare bot height to target height, penalize the difference between the two. Returns a positive magnitude and rew_scale_height is negative, so this deducts. Formula: (current height - target height)²

        + rew_scale_pose_match * pose_match_reward(leg_pos_offset) # ADDS points while the 8 leg joints as a whole stay within tolerance of the default pose, nothing once the bot folds away from it. Compares current pose to default pose. Formula: 1.0 if ||joint_pos - default_joint_pos|| <= tolerance else 0.0

        #Adding points
        + rew_scale_track_lin * track_lin_vel_xy(lin_vel_b, commands, tracking_sigma_lin) # tracking linear velocity x and y against the command x and y, using Gaussian, the closer both axes match the command the higher points it will add. Formula: exp(-((command x - current linear velocity x)² + (command y - current linear velocity y)²) / σ_linear). y is commanded now because the legs are free, so HAA can push the bot sideways
        + rew_scale_track_ang * track_ang_vel_z(ang_vel_b, commands, tracking_sigma_ang) # tracking angular velocity z axis and the command z axis, using Gaussian, the closer the z axis matches from the command z axis the higher points it will add. Formula: exp(-((current angular velocity z - command z axis)²) / σ_angular)

        # + rew_scale_stance_width * stance_width_reward(stance_width, stance_width_target, stance_width_tolerance) # ADDS points while the two wheels stay within tolerance of the nominal track width, nothing once they splay past it. Formula: 1.0 if |wheel separation - target| <= tolerance else 0.0
        
        + rew_scale_alive * alive_bonus(actions.shape[0], actions.device) # if the bot hasn't been terminated one way or another, it will keep giving 1 point to score to all. Formula: torch.one(env, device)
    )
    return total