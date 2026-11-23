"""
The ball center must cross the hoop plane downward within the collision-free
inner opening. This is a simplified 3D rim, not a complete basketball basket.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .basketball_throw_env import BasketballThrowEnv, ThrowConfig


@dataclass
class HoopConfig(ThrowConfig):
    hoop_radius: float = 0.16       # rim centerline radius, meters
    rim_tube_radius: float = 0.012
    ball_radius: float = 0.045
    success_margin: float = 0.005


class BasketballHoopEnv(BasketballThrowEnv):
    """Same 17D observation and 2D action as Phase 2; new objective."""
    def __init__(self, config=None, render_mode=None):
        cfg = config or HoopConfig()
        xml = Path(__file__).resolve().parents[1] / 'assets' / 'planar_hoop.xml'
        super().__init__(config=cfg, render_mode=render_mode, xml_path=xml)
        self.hoop_cfg = cfg
        self.hoop_success = False
        self.min_hoop_error = float('inf')

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.hoop_success = False
        self.min_hoop_error = float('inf')
        return obs, info

    @staticmethod
    def _downward_crossing(prev, curr, center, radius):
        """Return (crossed, xy_error) for ball-center downward plane crossing."""
        z = float(center[2])
        if not (prev[2] >= z > curr[2]):
            return False, float('inf')
        alpha = (prev[2] - z) / (prev[2] - curr[2])
        xy = prev[:2] + alpha * (curr[:2] - prev[:2])
        error = float(np.linalg.norm(xy - center[:2]))
        return error <= radius, error

    def step(self, action):
        prev_ball = self._ball_position()
        obs, _, _, truncated, info = super().step(action)
        ball = self._ball_position()

        entry_radius = (self.hoop_cfg.hoop_radius -
                        self.hoop_cfg.rim_tube_radius -
                        self.hoop_cfg.ball_radius -
                        self.hoop_cfg.success_margin)

        # The released ball must cross the rim plane downward inside the opening.
        success = False
        entry_error = float('inf')
        if self.released:
            success, entry_error = self._downward_crossing(
                prev_ball, ball, self.target, entry_radius
            )
            self.min_hoop_error = min(self.min_hoop_error, entry_error)

        self.hoop_success = success
        out_of_bounds = bool(ball[2] < 0.045 or ball[0] > 3.5 or abs(ball[1]) > 1.0)
        terminated = bool(success or out_of_bounds)
        reward = 0.0
        if success:
            reward = 1.0
        elif terminated or truncated:
            # best_distance remains infinite until the ball is released.
            distance = info['best_distance'] if self.released else info['distance']
            reward = -max(100.0 * distance, 1e-6)

        info.update(
            is_success=success,
            hoop_success=success,
            hoop_crossed=success,
            hoop_entry_passed=success,
            hoop_crossing_error=entry_error,
            min_hoop_error=float(self.min_hoop_error),
            hoop_clearance=entry_radius,
        )
        return obs, reward, terminated, truncated, info
