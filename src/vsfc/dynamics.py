"""Quadrotor rigid-body dynamics, paper eq. (1).

    p_dot = v
    m v_dot = -f_T R e3 + m g e3
    R_dot = R sk(omega)
    J omega_dot = -omega x J omega + tau

Frames (docs/equation_map.md §1): p, v in the inertial frame eF with e3 pointing down (A12);
R maps body -> inertial; omega, tau in the body frame.

State layout [Eng]: x = [p(3), v(3), R.flatten() (9, row-major), omega(3)], length 18.
R is integrated directly and re-projected onto SO(3) after each step (`project_to_so3`).
Euler angles are ZYX (A13), used only for output and references.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])
STATE_DIM = 18


def skew(w: np.ndarray) -> np.ndarray:
    """sk(w): (3,3) skew-symmetric matrix with sk(w) @ b == cross(w, b)."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def euler_to_rotation(theta: np.ndarray) -> np.ndarray:
    """ZYX Euler angles [phi, theta, psi] -> R = Rz(psi) Ry(theta) Rx(phi) (body -> inertial)."""
    phi, th, psi = theta
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(th), np.sin(th)
    cp, sp = np.cos(psi), np.sin(psi)
    return np.array([
        [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
        [-st, ct * sf, ct * cf],
    ])


def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """Inverse of `euler_to_rotation`, valid for |theta| < pi/2."""
    th = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, th, psi])


def euler_rate_matrix(theta: np.ndarray) -> np.ndarray:
    """W(Theta) with Theta_dot = W(Theta) omega for ZYX angles (exact kinematics).

    The paper's F_j in eq. (33) uses Theta_dot = omega, i.e. W = I (small-angle, A13).
    """
    phi, th, _ = theta
    cf, sf = np.cos(phi), np.sin(phi)
    ct, tt = np.cos(th), np.tan(th)
    return np.array([[1.0, sf * tt, cf * tt],
                     [0.0, cf, -sf],
                     [0.0, sf / ct, cf / ct]])


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (Frobenius norm) via SVD, with det = +1."""
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


@dataclass(frozen=True)
class QuadrotorParams:
    """m: kg; J: (3,3) kg m^2; g: m/s^2."""

    m: float
    J: np.ndarray
    g: float

    @property
    def J_inv(self) -> np.ndarray:
        return np.linalg.inv(self.J)


def pack_state(p, v, R, omega) -> np.ndarray:
    return np.concatenate([np.asarray(p, float), np.asarray(v, float),
                           np.asarray(R, float).reshape(9), np.asarray(omega, float)])


def unpack_state(x: np.ndarray):
    """Return (p, v, R, omega) views of an 18-vector."""
    return x[0:3], x[3:6], x[6:15].reshape(3, 3), x[15:18]


def quadrotor_derivatives(x: np.ndarray, f_T: float, tau: np.ndarray, params: QuadrotorParams) -> np.ndarray:
    """Right-hand side of eq. (1). f_T: N (scalar); tau: (3,) N m in body frame."""
    _, v, R, omega = unpack_state(x)
    p_dot = v
    v_dot = -(f_T / params.m) * (R @ E3) + params.g * E3
    R_dot = R @ skew(omega)
    omega_dot = params.J_inv @ (-np.cross(omega, params.J @ omega) + np.asarray(tau, float))
    return np.concatenate([p_dot, v_dot, R_dot.reshape(9), omega_dot])


def normalize_state(x: np.ndarray) -> np.ndarray:
    """Post-step hook: re-project the R block onto SO(3)."""
    x = x.copy()
    x[6:15] = project_to_so3(x[6:15].reshape(3, 3)).reshape(9)
    return x


def thrust_vector(x: np.ndarray, f_T: float) -> np.ndarray:
    """mu_j = R_j f_Tj e3 (paper Sec. IV), shape (3,), N."""
    _, _, R, _ = unpack_state(x)
    return f_T * (R @ E3)
