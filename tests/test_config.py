"""Smoke tests: the YAML reproduces the numbers printed in paper Sec. VI."""

import numpy as np
import pytest

from vsfc.config import load_case_study


@pytest.fixture(scope="module")
def cs():
    return load_case_study()


def test_leader_matches_paper(cs):
    assert cs.leader.mass == 1.0
    np.testing.assert_allclose(cs.leader.sphere_offset, [0, 0, 1])
    # p1(t) = [2.5+0.4t, -1+0.5t, 0.4t]
    np.testing.assert_allclose(cs.leader.position(10.0), [6.5, 4.0, 4.0])


def test_followers_match_paper(cs):
    f2, f3 = cs.followers
    assert (f2.id, f3.id) == (2, 3)
    assert (f2.mass, f3.mass) == (1.3, 2.0)
    np.testing.assert_allclose(np.diag(f2.inertia), [0.0082, 0.0080, 0.0145])
    assert f2.focal_length == pytest.approx(3.2e-3)
    np.testing.assert_allclose(f2.p0, [0.4, -0.3, 0])
    np.testing.assert_allclose(f3.p0, [1.5, 1.5, 0])
    np.testing.assert_allclose(f2.euler0, [0, 0, 0.174])
    np.testing.assert_allclose(f2.delta, [4, 1, 1])
    np.testing.assert_allclose(f3.delta, [4, -1, 1])
    assert (f2.beta, f2.gamma) == (5.0, 1.0)
    assert (f2.observer.alpha, f2.observer.kappa, f2.observer.eps) == (5.0, 5.0, 0.5)
    assert (f3.observer.alpha, f3.observer.kappa, f3.observer.eps) == (5.0, 55.0, 0.5)
    np.testing.assert_allclose(f2.Q, 10 * np.eye(6))
    np.testing.assert_allclose(f2.R, 8 * np.eye(3))
    np.testing.assert_allclose(f2.S, 10 * np.eye(12))
    np.testing.assert_allclose(f2.D, 8 * np.eye(3))
    assert f2.tol_vs == f2.tol_att == 1e-5
    assert cs.dT == 0.05


def test_derived_quantities(cs):
    f2, f3 = cs.followers
    assert f2.x_star == 4.0
    np.testing.assert_allclose(f2.s_star, [1.0, 0.25, 0.25])
    np.testing.assert_allclose(f3.s_star, [1.0, -0.25, 0.25])


def test_observer_gain_condition_theorem1(cs):
    # Theorem 1: alpha * kappa > 1 / x*
    for f in cs.followers:
        assert f.observer.alpha * f.observer.kappa > 1.0 / f.x_star


def test_rejects_bad_values(tmp_path):
    import yaml

    from vsfc.config import DEFAULT_CONFIG

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["followers"][0]["beta"] = -1.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="beta"):
        load_case_study(bad)
