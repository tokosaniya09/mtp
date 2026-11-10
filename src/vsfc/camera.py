"""Camera geometry and sphere image-moment features, paper eqs. (2)-(4).

Default feature model [Eng] (Way 1, decision D19):  s_ij = vq_ij / x*_ij,
i.e. the relative position of the sphere centre divided by the desired depth. This is what the paper's
image dynamics s_dot = v_ij / x*_ij assume (docs/equation_map.md §2, A8).

Eq. (3) is also implemented (`moment_features`) using disc moments (sphere image treated as a circle
centred on the projected centre); with that model it reproduces Way 1 exactly, which the tests check.
The exact elliptical sphere image is NOT modelled (A8 accepted as an assumption, D19).

Frames: vq is expressed in the virtual camera frame vF_j, whose axes are parallel to the inertial frame
(the rotation between cF_j and vF_j is R_j); x is the depth along the optical axis. SI units throughout
(focal length in metres).
"""

from __future__ import annotations

import numpy as np


# ---------- geometry (unnumbered equations, Sec. II) ----------

def sphere_centre(p_i: np.ndarray, h_i: np.ndarray) -> np.ndarray:
    """q_i = p_i + h_i: sphere centre on the leader, inertial frame, (3,) m."""
    return np.asarray(p_i, float) + np.asarray(h_i, float)


def relative_position_virtual(q_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    """vq_ij = q_i - p_j = [x, y, z]: sphere centre relative to follower j in vF_j, (3,) m."""
    return np.asarray(q_i, float) - np.asarray(p_j, float)


def _check_in_front(vq: np.ndarray) -> None:
    if not vq[0] > 0.0:
        raise ValueError(f"sphere is not in front of the camera (depth x = {vq[0]} <= 0)")


# ---------- eq. (2): perspective projection ----------

def project_point(vq: np.ndarray, lam: float) -> np.ndarray:
    """Eq. (2): [u, n] = lam * [y, z] / x on the virtual image plane, (2,) m."""
    vq = np.asarray(vq, float)
    _check_in_front(vq)
    return lam * vq[1:3] / vq[0]


def point_interaction_matrix(vq: np.ndarray, lam: float) -> np.ndarray:
    """Derivative of eq. (2): [u_dot, n_dot] = L @ v_ij, with
    L = [[-u/x, lam/x, 0], [-n/x, 0, lam/x]], shape (2, 3)."""
    vq = np.asarray(vq, float)
    _check_in_front(vq)
    x = vq[0]
    u, n = project_point(vq, lam)
    return np.array([[-u / x, lam / x, 0.0],
                     [-n / x, 0.0, lam / x]])


# ---------- eq. (3): image-moment features ----------

def disc_image_moments(vq: np.ndarray, radius: float, lam: float) -> tuple[float, float, float]:
    """Moments (a = m00, m10, m01) of the sphere image approximated as a disc of radius lam*r/x
    centred on the projected sphere centre: a = pi (lam r / x)^2, m10 = a u, m01 = a n."""
    vq = np.asarray(vq, float)
    _check_in_front(vq)
    a = np.pi * (lam * radius / vq[0]) ** 2
    u, n = project_point(vq, lam)
    return a, a * u, a * n


def desired_area(x_star: float, radius: float, lam: float) -> float:
    """a*_ij: disc image area when the sphere is at depth x* (A9)."""
    return np.pi * (lam * radius / x_star) ** 2


def moment_features(a: float, m10: float, m01: float, a_star: float, lam: float) -> np.ndarray:
    """Eq. (3): s_x = sqrt(a*/a), s_y = s_x m10/(lam a), s_z = s_x m01/(lam a). Returns (3,)."""
    if not a > 0.0:
        raise ValueError(f"image area must be > 0, got {a}")
    s_x = np.sqrt(a_star / a)
    return np.array([s_x, s_x * m10 / (lam * a), s_x * m01 / (lam * a)])


# ---------- Way 1 (default) and eq. (4) ----------

def ideal_features(vq: np.ndarray, x_star: float) -> np.ndarray:
    """Way 1 [Eng, D19]: s_ij = vq_ij / x*_ij, (3,), dimensionless."""
    vq = np.asarray(vq, float)
    _check_in_front(vq)
    return vq / x_star


def feature_rate(v_ij: np.ndarray, x_star: float) -> np.ndarray:
    """Image dynamics (unnumbered, after eq. (3)): s_dot = v_ij / x*_ij, (3,) 1/s."""
    return np.asarray(v_ij, float) / x_star


def feature_error(s: np.ndarray, s_star: np.ndarray) -> np.ndarray:
    """Eq. (4): e_ij = s_ij - s*_ij, (3,)."""
    return np.asarray(s, float) - np.asarray(s_star, float)
