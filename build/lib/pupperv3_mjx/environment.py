from typing import Any, Dict, List, Sequence, Tuple

import jax
import mujoco
import numpy as np
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp

from pupperv3_mjx import domain_randomization, rewards, utils

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def body_names_to_body_ids(mj_model, body_names: List[str]) -> np.array:
    body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name)
        for body_name in body_names
    ]
    assert not any(id_ == -1 for id_ in body_ids), "Body not found."
    return np.array(body_ids)


def body_name_to_geom_ids(mj_model, body_name: str) -> np.array:
    body = mj_model.body(body_name)
    return body.geomadr + np.arange(np.squeeze(body.geomnum))


def body_names_to_geom_idss(mj_model, body_names: List[str]) -> np.array:
    return np.concatenate(list(body_name_to_geom_ids(mj_model, name) for name in body_names))


class PupperV3Env(PipelineEnv):
    """Environment for training the Pupper V3 quadruped joystick policy in MJX."""

    def __init__(
        self,
        path: str,
        reward_config: Dict,
        action_scale: float,
        observation_history: int,
        joint_lower_limits: List = [
            -1.220,
            -0.420,
            -2.790,
            -2.510,
            -3.140,
            -0.710,
            -1.220,
            -0.420,
            -2.790,
            -2.510,
            -3.140,
            -0.710,
        ],
        joint_upper_limits: List = [
            2.510,
            3.140,
            0.710,
            1.220,
            0.420,
            2.790,
            2.510,
            3.140,
            0.710,
            1.220,
            0.420,
            2.790,
        ],
        dof_damping: float = 0.25,
        position_control_kp: float = 5.0,
        start_position_config: domain_randomization.StartPositionRandomization = (
            domain_randomization.StartPositionRandomization(
                x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
            )
        ),
        foot_site_names: List[str] = [
            "leg_front_r_3_foot_site",
            "leg_front_l_3_foot_site",
            "leg_back_r_3_foot_site",
            "leg_back_l_3_foot_site",
        ],
        torso_name: str = "base_link",
        upper_leg_body_names: List[str] = [
            "leg_front_r_2",
            "leg_front_l_2",
            "leg_back_r_2",
            "leg_back_l_2",
        ],
        lower_leg_body_names: List[str] = [
            "leg_front_r_3",
            "leg_front_l_3",
            "leg_back_r_3",
            "leg_back_l_3",
        ],
        resample_velocity_step: int = 500,
        linear_velocity_x_range: Tuple = (-0.75, 0.75),
        linear_velocity_y_range: Tuple = (-0.5, 0.5),
        angular_velocity_range: Tuple = (-2.0, 2.0),
        default_pose: jp.array = jp.array(
            [0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]
        ),
        angular_velocity_noise: float = 0.3,
        gravity_noise: float = 0.1,
        motor_angle_noise: float = 0.1,
        last_action_noise: float = 0.01,
        kick_vel: float = 0.2,
        kick_probability: float = 0.02,
        terminal_body_z: float = 0.1,
        early_termination_step_threshold: int = 500,
        terminal_body_angle: float = 0.52,
        foot_radius: float = 0.02,
        environment_timestep: float = 0.02,
        physics_timestep: float = 0.004,
        latency_distribution: jax.Array = jp.array([0.2, 0.8]),
        desired_world_z_in_body_frame: jax.Array = jp.array([0.0, 0.0, 1.0]),
        use_imu: bool = True,
    ):
        """
        Args:
            path (str): The path to the MJCF file.
            reward_config (Dict): The reward configuration.
            action_scale (float): The scale to apply to actions.
            observation_history (int): The number of previous observations to include in the state.
            joint_lower_limits (List): The lower limits for the joint angles.
            joint_upper_limits (List): The upper limits for the joint angles.
            dof_damping (float): The damping to apply to the DOFs.
            position_control_kp (float): The position control kp.
            start_position_config (domain_randomization.StartPositionRandomization):
            The start position randomization config.
            foot_site_names (List[str]): The names of the foot sites.
            torso_name (str): The name of the torso.
            upper_leg_body_names (List[str]): The names of the upper leg bodies.
            lower_leg_body_names (List[str]): The names of the lower leg bodies.
            resample_velocity_step (int): The number of steps to resample the velocity.
            linear_velocity_x_range (Tuple): The range of linear velocity in the x-direction.
            linear_velocity_y_range (Tuple): The range of linear velocity in the y-direction.
            angular_velocity_range (Tuple): The range of angular velocity.
            default_pose (jp.array): The default pose.
            angular_velocity_noise (float): The angular velocity noise.
            gravity_noise (float): The gravity noise.
            motor_angle_noise (float): The motor angle noise.
            last_action_noise (float): The last action noise.
            kick_vel (float): The kick velocity.
            kick_probability (float): The kick probability.
            terminal_body_z (float): The terminal body z.
            early_termination_step_threshold (int): The early termination step threshold.
            terminal_body_angle (float): The terminal body angle.
            foot_radius (float): The foot radius.
            environment_timestep (float): The environment timestep.
            physics_timestep (float): The physics timestep.
            latency_distribution (jax.Array): Probability distribution for action latency.
            First element corresponds to 0 latency. Shape: (N, 1)
            desired_world_z_in_body_frame (jax.Array): The desired world z in body frame.
            use_imu (bool): Whether to use IMU.
        """
        sys = mjcf.load(path)
        self._dt = environment_timestep  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": physics_timestep})

        # override menagerie params for smoother policy
        sys = sys.replace(
            # dof_damping=sys.dof_damping.at[6:].set(DOF_DAMPING),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(position_control_kp),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1]
            .set(-position_control_kp)
            .at[:, 2]
            .set(-dof_damping),
        )

        # override the default joint angles with default_pose
        sys.mj_model.keyframe("home").qpos[7:] = default_pose

        n_frames = self._dt // sys.opt.timestep
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._reward_config = reward_config
        self._torso_geom_ids = body_name_to_geom_ids(sys.mj_model, torso_name)
        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, torso_name
        )
        assert self._torso_idx != -1, "torso not found"
        self._action_scale = action_scale
        self._angular_velocity_noise = angular_velocity_noise
        self._gravity_noise = gravity_noise
        self._motor_angle_noise = motor_angle_noise
        self._last_action_noise = last_action_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = default_pose
        self.lowers = joint_lower_limits
        self.uppers = joint_upper_limits
        feet_site = foot_site_names
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)

        self._lower_leg_body_id = body_names_to_body_ids(sys.mj_model, lower_leg_body_names)
        self._upper_leg_geom_ids = body_names_to_geom_idss(sys.mj_model, upper_leg_body_names)

        self._foot_radius = foot_radius
        self._nv = sys.nv

        # start pos randomization params
        self._start_position_config = start_position_config

        # training params
        self._linear_velocity_x_range = linear_velocity_x_range
        self._linear_velocity_y_range = linear_velocity_y_range
        self._angular_velocity_range = angular_velocity_range

        self._kick_probability = kick_probability
        self._resample_velocity_step = resample_velocity_step

        # observation configuration
        self.observation_dim = 33
        self._observation_history = observation_history

        # reward configuration
        self._early_termination_step_threshold = early_termination_step_threshold

        # terminal condition
        self._terminal_body_z = terminal_body_z
        self._terminal_body_angle = terminal_body_angle

        # desired orientation
        self._desired_world_z_in_body_frame = jp.array(desired_world_z_in_body_frame)

        # latency
        self._latency_distribution = latency_distribution

        # whether to use imu
        self._use_imu = use_imu

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = self._linear_velocity_x_range  # min max [m/s]
        lin_vel_y = self._linear_velocity_y_range  # min max [m/s]
        ang_vel_yaw = self._angular_velocity_range  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, sample_command_key, randomize_pos_key = jax.random.split(rng, 3)

        init_q = domain_randomization.randomize_qpos(
            self._init_q, self._start_position_config, rng=randomize_pos_key
        )

        pipeline_state = self.pipeline_init(init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12, dtype=float),
            "action_buffer": jp.zeros((12, self._latency_distribution.shape[0]), dtype=float),
            "last_vel": jp.zeros(12, dtype=float),
            "command": self.sample_command(sample_command_key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4, dtype=float),
            "rewards": {k: 0.0 for k in self._reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "desired_world_z_in_body_frame": self._desired_world_z_in_body_frame,
        }

        obs_history = jp.zeros(
            self._observation_history * self.observation_dim, dtype=float
        )  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2, dtype=float)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types

        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2, kick_bernoulli, latency_key = jax.random.split(
            state.info["rng"], 5
        )
        state.info["rng"] = rng

        # Whether to kick and the kick velocity are both random
        kick = jax.random.uniform(kick_noise_2, shape=(2,), minval=-1, maxval=1) * self._kick_vel
        kick *= jax.random.bernoulli(kick_bernoulli, p=self._kick_probability, shape=(2,))
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # Handle action latency
        # Put the current action at the end of the action buffer
        # The action to take will be the first element of the buffer
        state.info["action_buffer"] = utils.circular_buffer_push_back(
            state.info["action_buffer"], action
        )

        # Pick from the action buffer by sampling from discrete distribution
        # Note that actions may be acted on out of order if the buffer is more than 2 elements
        action_buffer_idx = jax.random.choice(
            latency_key, self._latency_distribution.shape[0], p=self._latency_distribution
        )
        action_to_take = state.info["action_buffer"][:, action_buffer_idx]

        # Physics step
        motor_targets = self._default_pose + action_to_take * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # Foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # Done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < np.cos(
            self._terminal_body_angle
        )
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._terminal_body_z

        # Reward
        rewards_dict = {
            "tracking_lin_vel": rewards.reward_tracking_lin_vel(
                state.info["command"],
                x,
                xd,
                tracking_sigma=self._reward_config.rewards.tracking_sigma,
            ),
            "tracking_ang_vel": rewards.reward_tracking_ang_vel(
                state.info["command"],
                x,
                xd,
                tracking_sigma=self._reward_config.rewards.tracking_sigma,
            ),
            "tracking_orientation": rewards.reward_tracking_orientation(
                state.info["desired_world_z_in_body_frame"],
                x,
                tracking_sigma=self._reward_config.rewards.tracking_sigma,
            ),
            "lin_vel_z": rewards.reward_lin_vel_z(xd),
            "ang_vel_xy": rewards.reward_ang_vel_xy(xd),
            "orientation": rewards.reward_orientation(x),
            "torques": rewards.reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "joint_acceleration": rewards.reward_joint_acceleration(
                joint_vel, state.info["last_vel"], dt=self._dt
            ),
            "mechanical_work": rewards.reward_mechanical_work(
                pipeline_state.qfrc_actuator[6:], pipeline_state.qvel[6:]
            ),
            "action_rate": rewards.reward_action_rate(action, state.info["last_act"]),
            "stand_still": rewards.reward_stand_still(
                state.info["command"], joint_angles, self._default_pose
            ),
            "feet_air_time": rewards.reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": rewards.reward_foot_slip(
                pipeline_state,
                contact_filt_cm,
                feet_site_id=self._feet_site_id,
                lower_leg_body_id=self._lower_leg_body_id,
            ),
            "termination": rewards.reward_termination(
                done,
                state.info["step"],
                step_threshold=self._early_termination_step_threshold,
            ),
            "knee_collision": rewards.reward_geom_collision(
                pipeline_state, self._upper_leg_geom_ids
            ),
            "body_collision": rewards.reward_geom_collision(pipeline_state, self._torso_geom_ids),
        }
        rewards_dict = {
            k: v * self._reward_config.rewards.scales[k] for k, v in rewards_dict.items()
        }
        reward = jp.clip(sum(rewards_dict.values()) * self.dt, 0.0, 10000.0)

        # State management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards_dict
        state.info["step"] += 1

        # Sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # Reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._resample_velocity_step),
            0,
            state.info["step"],
        )

        # Log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        if self._use_imu:
            inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
            local_body_angular_velocity = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        else:
            inv_torso_rot = jp.array([1, 0, 0, 0])
            local_body_angular_velocity = jp.zeros(3)

        # See https://arxiv.org/abs/2202.05481 as reference for noise addition
        rng, ang_key, gravity_key, motor_angle_key, last_action_key = jax.random.split(
            state_info["rng"], 5
        )
        state_info["rng"] = rng

        ang_vel_noise = (
            jax.random.uniform(ang_key, (3,), minval=-1, maxval=1) * self._angular_velocity_noise
        )
        gravity_noise = (
            jax.random.uniform(gravity_key, (3,), minval=-1, maxval=1) * self._gravity_noise
        )
        motor_ang_noise = (
            jax.random.uniform(motor_angle_key, (12,), minval=-1, maxval=1)
            * self._motor_angle_noise
        )
        last_action_noise = (
            jax.random.uniform(last_action_key, (12,), minval=-1, maxval=1)
            * self._last_action_noise
        )

        noised_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot) + gravity_noise
        noised_gravity = noised_gravity / jp.linalg.norm(noised_gravity)

        # Construct observation and add noise
        obs = jp.concatenate(
            [
                local_body_angular_velocity + ang_vel_noise,  # angular velocity
                noised_gravity,  # gravity vector
                state_info["command"],  # command
                pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # motor angles
                state_info["last_act"] + last_action_noise,  # last action
            ]
        )

        assert self.observation_dim == obs.shape[0]

        # clip
        obs = jp.clip(obs, -100.0, 100.0)

        # stack observations through time
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history

    def render(
        self, trajectory: List[base.State], camera: str | None = None
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera)
