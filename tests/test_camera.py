"""Tests for eqs. (2)-(4): src/vsfc/camera.py."""

import numpy as np
import pytest

from vsfc.camera import (desired_area, disc_image_moments, feature_error, feature_rate, ideal_features,
                         moment_features, point_interaction_matrix, project_point, relative_position_virtual,
                         sphere_centre)
from vsfc.config import load_case_study

LAM = 3.2e-3     # paper Sec. VI focal length
RADIUS = 0.1     # [Eng] A9
X_STAR = 4.0     # A9/A10


@pytest.fixture(scope="module")
def cs():
    return load_case_study()


def test_worked_example_follower2_t0(cs):
    # Hand calculation shown to the user: q1 = [2.5,-1,1], p2 = [0.4,-0.3,0].
    f2 = cs.followers[0]
    q1 = sphere_centre(cs.leader.position(0.0), cs.leader.sphere_offset)
    vq = relative_position_virtual(q1, f2.p0)
    np.testing.assert_allclose(vq, [2.1, -0.7, 1.0])
    s = ideal_features(vq, f2.x_star)
    np.testing.assert_allclose(s, [0.525, -0.175, 0.25])
    np.testing.assert_allclose(feature_error(s, f2.s_star), [-0.475, -0.425, 0.0], atol=1e-15)


def test_worked_example_image_values():
    vq = np.array([2.1, -0.7, 1.0])
    a, m10, m01 = disc_image_moments(vq, RADIUS, LAM)
    assert a * 1e6 == pytest.approx(0.0730, abs=5e-4)                          # mm^2
    assert desired_area(X_STAR, RADIUS, LAM) * 1e6 == pytest.approx(0.0201, abs=5e-4)
    np.testing.assert_allclose(project_point(vq, LAM) * 1e3, [-1.0667, 1.5238], atol=1e-4)  # mm


def test_error_zero_at_desired_position(cs):
    for f in cs.followers:
        np.testing.assert_allclose(feature_error(ideal_features(f.delta, f.x_star), f.s_star), 0.0)


def test_eq3_disc_moments_equal_way1():
    rng = np.random.default_rng(0)
    a_star = desired_area(X_STAR, RADIUS, LAM)
    for _ in range(200):
        vq = rng.uniform([0.5, -3, -3], [10, 3, 3])
        s_eq3 = moment_features(*disc_image_moments(vq, RADIUS, LAM), a_star, LAM)
        np.testing.assert_allclose(s_eq3, ideal_features(vq, X_STAR), rtol=1e-12)


def test_feature_rate_matches_finite_difference():
    rng = np.random.default_rng(1)
    vq, v = rng.uniform([1, -2, -2], [8, 2, 2]), rng.normal(size=3)
    dt = 1e-6
    fd = (ideal_features(vq + v * dt, X_STAR) - ideal_features(vq, X_STAR)) / dt
    np.testing.assert_allclose(feature_rate(v, X_STAR), fd, rtol=1e-6)


def test_interaction_matrix_matches_finite_difference():
    rng = np.random.default_rng(2)
    vq, v = rng.uniform([1, -2, -2], [8, 2, 2]), rng.normal(size=3)
    dt = 1e-7
    fd = (project_point(vq + v * dt, LAM) - project_point(vq, LAM)) / dt
    np.testing.assert_allclose(point_interaction_matrix(vq, LAM) @ v, fd, rtol=1e-5, atol=1e-12)


def test_rejects_sphere_behind_camera():
    for fn in (lambda q: project_point(q, LAM), lambda q: ideal_features(q, X_STAR),
               lambda q: disc_image_moments(q, RADIUS, LAM)):
        with pytest.raises(ValueError, match="in front"):
            fn(np.array([-1.0, 0.0, 0.0]))
