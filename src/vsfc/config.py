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
class OuterDataConfig:
    """[Eng, D26] data collection for Alg. 3, outer loop (Phase 1.5). See `sim/outer_loop.collect_dataset`."""

    n_runs: int
    run_length: float          # s
    burn_in: float             # s, dropped from the start of each run
    k_e: float                 # behaviour PD gain on e_k, N
    k_v: float                 # behaviour PD gain on v_hat_k, N s/m
    explore_freqs: np.ndarray  # (Q,) rad/s
    explore_amplitude: np.ndarray  # (Q, 3) N
    bias_low: np.ndarray       # (3,) N
    bias_high: np.ndarray      # (3,) N
    e0_low: np.ndarray         # (3,)
    e0_high: np.ndarray        # (3,)
    v0_max: float              # m/s
    seed: int


@dataclass(frozen=True)
class OuterLearningConfig:
    """[Eng, D25-D28] Alg. 3 settings for the visual-servoing loop."""

    critic_degrees: tuple
    actor_degrees: tuple
    max_iter: int
    init_seeds: tuple
    init_gain_range: tuple      # (2,) N, positive
    init_cubic_std: float
    data: OuterDataConfig
    large_error_e0: np.ndarray  # (3,)
    near_saturation_v0: np.ndarray  # (3,) m/s


def _outer_learning(d: dict) -> OuterLearningConfig:
    dd = d["data"]
    freqs = np.asarray(dd["explore_freqs"], float)
    amp = np.asarray(dd["explore_amplitude"], float)
    if amp.shape != (freqs.size, 3):
        raise ValueError(f"explore_amplitude: expected ({freqs.size}, 3), got {amp.shape}")
    lo, hi = _vec(dd["bias_low"], 3, "bias_low"), _vec(dd["bias_high"], 3, "bias_high")
    e_lo, e_hi = _vec(dd["e0_low"], 3, "e0_low"), _vec(dd["e0_high"], 3, "e0_high")
    if np.any(lo > hi) or np.any(e_lo > e_hi):
        raise ValueError("low bounds must not exceed high bounds")
    data = OuterDataConfig(
        n_runs=int(dd["n_runs"]), run_length=_positive(dd["run_length"], "run_length"),
        burn_in=float(dd["burn_in"]), k_e=_positive(dd["k_e"], "k_e"), k_v=_positive(dd["k_v"], "k_v"),
        explore_freqs=freqs, explore_amplitude=amp, bias_low=lo, bias_high=hi, e0_low=e_lo, e0_high=e_hi,
        v0_max=float(dd["v0_max"]), seed=int(dd["seed"]))
    if data.burn_in < 0 or data.burn_in >= data.run_length or data.n_runs < 1:
        raise ValueError("need 0 <= burn_in < run_length and n_runs >= 1")
    return OuterLearningConfig(
        critic_degrees=tuple(int(x) for x in d["critic_degrees"]),
        actor_degrees=tuple(int(x) for x in d["actor_degrees"]),
        max_iter=int(d["max_iter"]), init_seeds=tuple(int(x) for x in d["init_seeds"]),
        init_gain_range=tuple(_positive(x, "init_gain_range") for x in d["init_gain_range"]),
        init_cubic_std=float(d["init_cubic_std"]), data=data,
        large_error_e0=_vec(d["large_error_e0"], 3, "large_error_e0"),
        near_saturation_v0=_vec(d["near_saturation_v0"], 3, "near_saturation_v0"))


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
    ref_rate_tau: float
    sphere_radius: float
    observer_init: dict
    learning_outer: OuterLearningConfig


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
        ref_rate_tau=_positive(eng["ref_rate_tau"], "ref_rate_tau"),
        sphere_radius=_positive(eng["sphere_radius"], "sphere_radius"),
        observer_init=dict(eng["observer_init"]),
        learning_outer=_outer_learning(raw["learning_outer"]),
    )
