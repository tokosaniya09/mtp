"""Load the case-study configuration (paper Sec. VI) into typed, validated objects.

Source of values: config/paper_case_study.yaml, where each value is annotated [Paper] or [Eng].
This module only parses and validates; it adds no defaults of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "paper_case_study.yaml"


def _vec(values, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"{name}: expected shape ({n},), got {arr.shape}")
    return arr


def _positive(value, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name}: must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class ObserverGains:
    """Gains of the adaptive velocity observer, eq. (6)."""

    alpha: float
    kappa: float
    eps: float


@dataclass(frozen=True)
class LeaderConfig:
    id: int
    mass: float
    inertia: np.ndarray        # (3,3) kg m^2
    sphere_offset: np.ndarray  # (3,) m, h_i
    p0: np.ndarray             # (3,) m
    velocity: np.ndarray       # (3,) m/s, constant

    def position(self, t: float) -> np.ndarray:
        """Leader trajectory p_1(t) = p0 + velocity * t, paper Sec. VI."""
        return self.p0 + self.velocity * t


@dataclass(frozen=True)
class FollowerConfig:
    id: int
    mass: float
    inertia: np.ndarray        # (3,3) kg m^2
    focal_length: float        # m
    p0: np.ndarray             # (3,) m
    euler0: np.ndarray         # (3,) rad, ZYX
    v0: np.ndarray             # (3,) m/s
    omega0: np.ndarray         # (3,) rad/s
    delta: np.ndarray          # (3,) m, desired q_i - p_j (A10)
    observer: ObserverGains
    beta: float
    gamma: float
    Q: np.ndarray              # (6,6)
    R: np.ndarray              # (3,3)
    S: np.ndarray              # (12,12)
    D: np.ndarray              # (3,3)
    tol_vs: float
    tol_att: float

    @property
    def x_star(self) -> float:
        """Desired depth x*_ij = Delta_x (A9, A10)."""
        return float(self.delta[0])

    @property
    def s_star(self) -> np.ndarray:
        """Desired image feature s*_ij = Delta_ij / x*_ij (A10)."""
        return self.delta / self.x_star


@dataclass(frozen=True)
class CaseStudy:
    g: float
    leader: LeaderConfig
    followers: tuple[FollowerConfig, ...]
    dT: float
    duration: float
    step: float
    integrator: str
    gravity_compensation: bool
    e_dot_source: str
    psi_ref: float
    sphere_radius: float
    observer_init: dict


def _follower(d: dict) -> FollowerConfig:
    fid = int(d["id"])
    obs = d["observer"]
    f = FollowerConfig(
        id=fid,
        mass=_positive(d["mass"], f"follower {fid} mass"),
        inertia=np.diag(_vec(d["inertia_diag"], 3, "inertia_diag")),
        focal_length=_positive(d["focal_length"], "focal_length"),
        p0=_vec(d["p0"], 3, "p0"),
        euler0=_vec(d["euler0"], 3, "euler0"),
        v0=_vec(d["v0"], 3, "v0"),
        omega0=_vec(d["omega0"], 3, "omega0"),
        delta=_vec(d["delta"], 3, "delta"),
        observer=ObserverGains(
            alpha=_positive(obs["alpha"], "alpha"),
            kappa=_positive(obs["kappa"], "kappa"),
            eps=_positive(obs["eps"], "eps"),
        ),
        beta=_positive(d["beta"], "beta"),
        gamma=_positive(d["gamma"], "gamma"),
        Q=_positive(d["Q_scale"], "Q_scale") * np.eye(6),
        R=_positive(d["R_scale"], "R_scale") * np.eye(3),
        S=_positive(d["S_scale"], "S_scale") * np.eye(12),
        D=_positive(d["D_scale"], "D_scale") * np.eye(3),
        tol_vs=_positive(d["tol_vs"], "tol_vs"),
        tol_att=_positive(d["tol_att"], "tol_att"),
    )
    if f.x_star <= 0.0:
        raise ValueError(f"follower {fid}: x* = Delta_x must be > 0 (leader in front of camera)")
    return f


def load_case_study(path: str | Path = DEFAULT_CONFIG) -> CaseStudy:
    """Parse the YAML case study into a validated CaseStudy."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    ld = raw["leader"]
    leader = LeaderConfig(
        id=int(ld["id"]),
        mass=_positive(ld["mass"], "leader mass"),
        inertia=np.diag(_vec(ld["inertia_diag"], 3, "leader inertia_diag")),
        sphere_offset=_vec(ld["sphere_offset"], 3, "sphere_offset"),
        p0=_vec(ld["trajectory"]["p0"], 3, "leader p0"),
        velocity=_vec(ld["trajectory"]["velocity"], 3, "leader velocity"),
    )
    eng = raw["engineering"]
    if eng["e_dot_source"] not in ("exact", "finite_difference"):
        raise ValueError(f"e_dot_source must be 'exact' or 'finite_difference', got {eng['e_dot_source']!r}")
    sim = raw["simulation"]
    if sim["integrator"] != "rk4":
        raise ValueError(f"only 'rk4' is implemented, got {sim['integrator']!r}")

    return CaseStudy(
        g=_positive(raw["physics"]["g"], "g"),
        leader=leader,
        followers=tuple(_follower(f) for f in raw["followers"]),
        dT=_positive(raw["learning"]["dT"], "dT"),
        duration=_positive(sim["duration"], "duration"),
        step=_positive(sim["step"], "step"),
        integrator=sim["integrator"],
        gravity_compensation=bool(eng["gravity_compensation"]),
        e_dot_source=eng["e_dot_source"],
        psi_ref=float(eng["psi_ref"]),
        sphere_radius=_positive(eng["sphere_radius"], "sphere_radius"),
        observer_init=dict(eng["observer_init"]),
    )
