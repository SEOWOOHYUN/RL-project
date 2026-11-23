
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


@dataclass
class ThrowConfig:
    episode_time: float = 2.0
    control_dt: float = 0.02
    release_time: float = 0.50
    action_scale: float = 1.4
    success_radius: float = 0.15
    target: tuple = (1.25, 0.0, 0.90)


class BasketballThrowEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config=None, render_mode=None, xml_path=None):
        super().__init__()

        self.cfg = config or ThrowConfig()

        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")

        self.render_mode = render_mode

        path = (
            Path(xml_path)
            if xml_path is not None
            else Path(__file__).resolve().parents[1]
            / "assets" / "planar_throw.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = max(
            1,
            round(self.cfg.control_dt / self.model.opt.timestep),
        )
        self.dt = self.frame_skip * self.model.opt.timestep

        self.ball_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold_ball"
        )
        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "hand"
        )

        self.joints = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in ("shoulder", "elbow")
        ]

        self.qadr = np.array([
            self.model.jnt_qposadr[j] for j in self.joints
        ])
        self.vadr = np.array([
            self.model.jnt_dofadr[j] for j in self.joints
        ])

        self.ball_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.ball_qadr = self.model.jnt_qposadr[self.ball_joint]
        self.ball_vadr = self.model.jnt_dofadr[self.ball_joint]

        self.target = np.array(self.cfg.target, dtype=np.float64)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32,
        )

        self.renderer = None
        self.step_count = 0
        self.released = False
        self.best_dist = float("inf")
        self.prev_action = np.zeros(2, dtype=np.float32)

    def _ball_position(self):
        return self.data.xpos[self.ball_id].copy()

    def _ball_velocity(self):
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ball_id,
            vel,
            0,
        )
        return vel[3:].copy()

    def _obs(self):
        return np.concatenate((
            self.data.qpos[self.qadr],
            self.data.qvel[self.vadr],
            self._ball_position(),
            self._ball_velocity(),
            self.target,
            self.prev_action,
            [float(self.released)],
            [max(
                0.0,
                self.cfg.episode_time - self.step_count * self.dt,
            )],
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.model.body_pos[self.target_id] = self.target

        # XML defines vertical upward as qpos = [0, 0].
        # Keep the initial state fixed to prevent reward exploitation.
        self.data.qpos[self.qadr] = [-np.pi / 2, 0.0]
        self.data.qvel[self.vadr] = [0.0, 0.0]

        # Initialize position actuator targets at the same pose.
        self.data.ctrl[:] = [-np.pi / 2, 0.0]

        # Initialize ball quaternion and velocity.
        self.data.qpos[
            self.ball_qadr + 3:self.ball_qadr + 7
        ] = [1.0, 0.0, 0.0, 0.0]

        self.data.qvel[
            self.ball_vadr:self.ball_vadr + 6
        ] = 0.0

        self.data.eq_active[self.weld_id] = 1

        # Compute the hand pose at the vertical configuration.
        mujoco.mj_forward(self.model, self.data)

        hand_position = self.data.site_xpos[self.hand_id].copy()
        hand_rotation = self.data.site_xmat[
            self.hand_id
        ].reshape(3, 3)

        # Weld offset = 0.4 m in link2's local X direction.
        # The hand site is at 0.35 m, so add 0.05 m.
        ball_position = (
            hand_position
            + hand_rotation @ np.array([0.05, 0.0, 0.0])
        )

        self.data.qpos[
            self.ball_qadr:self.ball_qadr + 3
        ] = ball_position

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.released = False
        self.best_dist = float("inf")
        self.prev_action[:] = 0.0

        return self._obs(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        # Position control, relative to vertical initial pose.
        self.initial_qpos = np.array(
            [-np.pi / 2, 0.0],
            dtype=np.float64
        )

        # Relative position control
        target_qpos = (
            self.initial_qpos
            + self.cfg.action_scale * action
        )

        self.data.ctrl[:] = np.clip(
            target_qpos,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        # Fixed release at 0.5 s.
        if (
            not self.released
            and self.step_count * self.dt >= self.cfg.release_time
        ):
            self.data.eq_active[self.weld_id] = 0
            self.released = True

        prev_dist = float(np.linalg.norm(
            self._ball_position() - self.target
        ))

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        ball = self._ball_position()
        dist = float(np.linalg.norm(ball - self.target))

        if self.released:
            self.best_dist = min(self.best_dist, dist)

        success = bool(
            self.released
            and dist < self.cfg.success_radius
        )

        # No proximity reward while the ball is held.
        if self.released:
            reward = (
                -0.02 * dist
                + 0.4 * (prev_dist - dist)
            )
        else:
            reward = 0.0

        reward -= 0.001 * float(np.sum(action ** 2))

        if success:
            reward += 10.0

        ball_out_of_bounds = bool( ball[2] < 0.045 or ball[0] > 3.5 or abs(ball[1]) > 1.0)

        terminated = bool(success or (self.released and ball_out_of_bounds))

        truncated = bool(
            self.step_count * self.dt >= self.cfg.episode_time
        )

        self.prev_action = action.copy()

        info = {
            "distance": dist,
            "best_distance": float(self.best_dist),
            "is_success": success,
            "released": self.released,
            "elapsed_steps": self.step_count,
        }

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=480,
                width=640,
            )

        self.renderer.update_scene(
            self.data,
            camera="side",
        )
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
