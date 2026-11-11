"""Kinematic leader trajectories (A19: leader attitude not modelled; h_i constant in the inertial frame).

Each leader exposes `state(t) -> (p, v, a)` (inertial, (3,) each) and `accel_bound` (zeta_i, m/s^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstantVelocityLeader:
    """[Paper] Sec. VI: p_1(t) = p0 + velocity t. Acceleration is zero (A23)."""

    p0: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_config(cls, leader) -> "ConstantVelocityLeader":
        return cls(p0=leader.p0, velocity=leader.velocity)

    @property
    def accel_bound(self) -> float:
        return 0.0

    def state(self, t: float):
        return self.p0 + self.velocity * t, self.velocity.copy(), np.zeros(3)


@dataclass(frozen=True)
class SinusoidalLeader:
    """[Eng] test trajectory, not in the paper: p(t) = p0 + velocity t + amp * sin(omega t).

    Exercises the observer's adaptive term (A23). ||v_dot|| <= ||amp|| omega^2.
    """

    p0: np.ndarray
    velocity: np.ndarray
    amp: np.ndarray
    omega: float

    @property
    def accel_bound(self) -> float:
        return float(np.linalg.norm(self.amp)) * self.omega ** 2

    def state(self, t: float):
        s, c = np.sin(self.omega * t), np.cos(self.omega * t)
        p = self.p0 + self.velocity * t + self.amp * s
        v = self.velocity + self.amp * self.omega * c
        a = -self.amp * self.omega ** 2 * s
        return p, v, a
