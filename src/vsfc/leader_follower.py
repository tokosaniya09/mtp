"""Image-based leader-follower dynamics, paper eq. (5).

    e_dot_ij = v_ij / x*_ij
    v_dot_ij = -g e3 + R_j f_Tj e3 / m_j + v_dot_i,      ||v_dot_i|| <= zeta_i

This is the dynamics of the RELATIVE state (the leader as seen by follower j's camera). The leader moves
freely; its acceleration v_dot_i is an unknown, bounded disturbance. The follower's thrust vector
mu_j = R_j f_Tj e3 is the only control input (docs/equation_map.md §3).

Derivation: v_ij = v_i - v_j, and eq. (1) gives v_dot_j = g e3 - R_j f_Tj e3 / m_j (e3 points down, A12),
so v_dot_ij = v_dot_i - g e3 + R_j f_Tj e3 / m_j. With Way 1 features (D19), s_ij = vq_ij / x*_ij and the
virtual frame is parallel to the inertial frame, so e_dot_ij = v_ij / x*_ij holds exactly.

`thrust_vec` here is the PHYSICAL thrust vector R_j f_Tj e3. Under D14 the controller's learned input is
mu = thrust_vec - m_j g e3; that mapping lives in the controller, not here.

This module is not integrated by the simulator (the plant is eq. (1) for each vehicle). It is the reference
model the observer (eq. (6)) is checked against.
"""

from __future__ import annotations

import numpy as np

from vsfc.camera import feature_error, ideal_features, relative_position_virtual, sphere_centre
from vsfc.dynamics import E3


def relative_state(p_i: np.ndarray, v_i: np.ndarray, h_i: np.ndarray, p_j: np.ndarray, v_j: np.ndarray,
                   x_star: float, s_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative state (e_ij, v_ij) from the two vehicles' inertial positions and velocities.

    e_ij = (q_i - p_j)/x* - s*  (eqs. (3)-(4), Way 1), v_ij = v_i - v_j (h_i constant in the inertial
    frame, A19, so q_dot_i = v_i). Shapes (3,), (3,); units -, m/s.
    """
    vq = relative_position_virtual(sphere_centre(p_i, h_i), p_j)
    e = feature_error(ideal_features(vq, x_star), s_star)
    v_ij = np.asarray(v_i, float) - np.asarray(v_j, float)
    return e, v_ij


def relative_dynamics(v_ij: np.ndarray, thrust_vec: np.ndarray, a_leader: np.ndarray,
                      m_j: float, g: float, x_star: float) -> tuple[np.ndarray, np.ndarray]:
    """Right-hand side of eq. (5). Returns (e_dot (3,) 1/s, v_ij_dot (3,) m/s^2).

    v_ij: (3,) m/s; thrust_vec: R_j f_Tj e3, (3,) N; a_leader: v_dot_i, (3,) m/s^2; m_j: kg; x_star: m.
    """
    e_dot = np.asarray(v_ij, float) / x_star
    v_ij_dot = -g * E3 + np.asarray(thrust_vec, float) / m_j + np.asarray(a_leader, float)
    return e_dot, v_ij_dot
