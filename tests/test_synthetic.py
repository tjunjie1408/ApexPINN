import jax
import numpy as np
import pytest

from apexpinn.physics.residuals import COMPOUNDS
from apexpinn.physics.synthetic import (
    DEFAULT_COMPOUNDS,
    MAX_LAPS,
    MIN_LAPS,
    MISSPEC_VARIANTS,
    RACE_LENGTH,
    TRAFFIC_MIN_START,
    Misspecification,
    SyntheticDataset,
    default_theta_true,
    make_synthetic_dataset,
)


@pytest.fixture(scope="module")
def default_dataset() -> SyntheticDataset:
    return make_synthetic_dataset(jax.random.key(0))


def _valid(ds: SyntheticDataset, name: str) -> list[np.ndarray]:
    values = np.asarray(getattr(ds, name))
    return [values[s, :n] for s, n in enumerate(np.asarray(ds.n_laps))]


def test_default_dataset_shapes_and_ranges(default_dataset):
    ds = default_dataset
    n_stints = ds.b.shape[0]
    n_laps = np.asarray(ds.n_laps)
    assert ds.h.shape == ds.w.shape == ds.y.shape == ds.y_clean.shape == (n_stints, MAX_LAPS)
    assert ds.q.shape == (n_stints, MAX_LAPS + 1)
    assert np.all((n_laps >= MIN_LAPS) & (n_laps <= MAX_LAPS))
    np.testing.assert_array_equal(np.asarray(ds.mask).sum(axis=1), n_laps)
    used = {COMPOUNDS[i] for i in np.asarray(ds.compound_idx)}
    assert used == set(DEFAULT_COMPOUNDS) and len(used) == 3
    q = np.asarray(ds.q)
    assert np.all((q >= 0.0) & (q <= 1.0))
    # Circuit-level q differs between stints, so the schedules vary by stint.
    assert np.std(q.mean(axis=1)) > 0.05


# Acceptance test 1
def test_true_h_stays_in_unit_interval(default_dataset):
    for h in _valid(default_dataset, "h"):
        assert np.all(h >= 0.0) and np.all(h <= 1.0)


# Acceptance test 2
def test_true_w_is_nondecreasing(default_dataset):
    for w in _valid(default_dataset, "w"):
        assert w[0] >= 0.0
        assert np.all(np.diff(w) >= 0.0)


# Acceptance test 3 (S4)
def test_most_default_stints_show_an_early_dip(default_dataset):
    ds = default_dataset
    shapes = [y - b for y, b in zip(_valid(ds, "y_clean"), np.asarray(ds.b), strict=True)]
    dips = [2 <= int(np.argmin(g)) + 1 <= 8 and g[-1] > g.min() for g in shapes]
    assert np.mean(dips) >= 0.5


# Acceptance test 4
def test_same_key_same_dataset_different_key_different_dataset():
    theta = default_theta_true()
    first = make_synthetic_dataset(jax.random.key(7), 12, DEFAULT_COMPOUNDS, theta, 0.05)
    again = make_synthetic_dataset(jax.random.key(7), 12, DEFAULT_COMPOUNDS, theta, 0.05)
    other = make_synthetic_dataset(jax.random.key(8), 12, DEFAULT_COMPOUNDS, theta, 0.05)

    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(again), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(first.y), np.asarray(other.y))
    assert not np.array_equal(np.asarray(first.b), np.asarray(other.b))


def test_noise_level_matches_noise_s():
    ds = make_synthetic_dataset(jax.random.key(1), 40, DEFAULT_COMPOUNDS, default_theta_true(), 0.15)
    mask = np.asarray(ds.mask)
    noise = (np.asarray(ds.y) - np.asarray(ds.y_clean))[mask]
    assert abs(noise.std() - 0.15) < 0.015
    noiseless = make_synthetic_dataset(jax.random.key(1), 4, DEFAULT_COMPOUNDS, default_theta_true(), 0.0)
    np.testing.assert_array_equal(np.asarray(noiseless.y), np.asarray(noiseless.y_clean))


def test_rejects_bad_arguments():
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="n_stints"):
        make_synthetic_dataset(key, 0)
    with pytest.raises(ValueError, match="C9"):
        make_synthetic_dataset(key, 3, ("C2", "C9"))
    with pytest.raises(ValueError, match="noise_s"):
        make_synthetic_dataset(key, 3, noise_s=-0.1)


# Misspecification variants (PLAN T3). Each one changes the data-generating process, not the fitted model.


def _assert_same_leaves(first: SyntheticDataset, second: SyntheticDataset) -> None:
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _excess_kurtosis(x: np.ndarray) -> float:
    z = (x - x.mean()) / x.std()
    return float(np.mean(z**4) - 3.0)


def test_well_specified_variant_matches_the_default_and_variants_share_its_draws():
    key = jax.random.key(13)
    default = make_synthetic_dataset(key, 12)
    _assert_same_leaves(default, make_synthetic_dataset(key, 12, misspec="well_specified"))
    np.testing.assert_array_equal(np.asarray(default.traffic), 0.0)
    np.testing.assert_array_equal(np.asarray(default.rho_target), np.asarray(default.theta_true.rho))
    shifted = make_synthetic_dataset(key, 12, misspec="wrong_phi")
    for name in ("n_laps", "compound_idx", "q", "b", "h", "w", "race_lap", "mask"):
        np.testing.assert_array_equal(np.asarray(getattr(shifted, name)), np.asarray(getattr(default, name)))


@pytest.mark.parametrize("name", sorted(MISSPEC_VARIANTS))
def test_each_variant_is_deterministic_per_key(name):
    first = make_synthetic_dataset(jax.random.key(11), 12, misspec=name)
    again = make_synthetic_dataset(jax.random.key(11), 12, misspec=MISSPEC_VARIANTS[name])
    other = make_synthetic_dataset(jax.random.key(12), 12, misspec=name)
    _assert_same_leaves(first, again)
    assert first.misspec == MISSPEC_VARIANTS[name]
    assert not np.array_equal(np.asarray(first.y), np.asarray(other.y))


def test_race_laps_are_consecutive_and_inside_the_race():
    ds = make_synthetic_dataset(jax.random.key(14), 12)
    for race_lap in _valid(ds, "race_lap"):
        assert race_lap[0] >= 2 and race_lap[-1] <= RACE_LENGTH
        np.testing.assert_array_equal(np.diff(race_lap), 1)


def test_wrong_phi_adds_a_trend_with_slope_phi_used_minus_phi_true():
    spec = MISSPEC_VARIANTS["wrong_phi"]
    key = jax.random.key(3)
    base = make_synthetic_dataset(key, 12, noise_s=0.0)
    wrong = make_synthetic_dataset(key, 12, noise_s=0.0, misspec=spec)
    leftovers = [a - b for a, b in zip(_valid(wrong, "y_clean"), _valid(base, "y_clean"), strict=True)]
    for extra, race_lap in zip(leftovers, _valid(wrong, "race_lap"), strict=True):
        expected = (spec.phi_true - spec.phi_used) * (RACE_LENGTH - race_lap)
        np.testing.assert_allclose(extra, expected, atol=1e-4)
        # Per stint lap the target rises by phi_used - phi_true, which a fit puts into rho_c (PROJECT L3).
        np.testing.assert_allclose(np.diff(extra), spec.rho_shift, atol=1e-4)
    assert spec.rho_shift == pytest.approx(0.01)
    np.testing.assert_allclose(
        np.asarray(wrong.rho_target), np.asarray(wrong.theta_true.rho) + spec.rho_shift, rtol=1e-6
    )


def test_student_t_noise_has_std_noise_s_and_heavier_tails_than_gaussian():
    key = jax.random.key(4)
    heavy = make_synthetic_dataset(key, 200, noise_s=0.1, misspec="student_t")
    gauss = make_synthetic_dataset(key, 200, noise_s=0.1)
    noise_t = (np.asarray(heavy.y) - np.asarray(heavy.y_clean))[np.asarray(heavy.mask)]
    noise_g = (np.asarray(gauss.y) - np.asarray(gauss.y_clean))[np.asarray(gauss.mask)]
    assert abs(noise_t.std() / 0.1 - 1.0) < 0.1
    assert _excess_kurtosis(noise_t) > 1.0
    assert abs(_excess_kurtosis(noise_g)) < 0.5
    np.testing.assert_array_equal(np.asarray(heavy.y_clean), np.asarray(gauss.y_clean))


def test_piecewise_wear_is_monotone_with_a_mid_stint_slope_change():
    spec = MISSPEC_VARIANTS["piecewise_wear"]
    ratio = spec.wear_slope_ratio
    ds = make_synthetic_dataset(jax.random.key(5), 12, noise_s=0.0, misspec=spec)
    rho = np.asarray(ds.theta_true.rho)[np.asarray(ds.compound_idx)]
    for s, w in enumerate(_valid(ds, "w")):
        assert w[0] > 0.0 and np.all(np.diff(w) > 0.0)
        first_slope, last_slope = w[0], w[-1] - w[-2]  # w(0) = 0, so w at tau = 1 is the first slope
        np.testing.assert_allclose(first_slope, 2.0 * rho[s] / (1.0 + ratio), rtol=1e-4)
        np.testing.assert_allclose(last_slope, ratio * first_slope, rtol=1e-4)
        # Truth target: the stint-mean wear slope equals rho_c.
        np.testing.assert_allclose(w[-1] / w.size, rho[s], rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(ds.rho_target), np.asarray(ds.theta_true.rho))


@pytest.mark.parametrize("drop", [1, 2])
def test_drop_first_laps_unscores_early_laps_but_keeps_their_values(drop):
    key = jax.random.key(6)
    base = make_synthetic_dataset(key, 12)
    ds = make_synthetic_dataset(key, 12, misspec=Misspecification(drop_first_laps=drop))
    mask, base_mask = np.asarray(ds.mask), np.asarray(base.mask)
    assert base_mask[:, 0].all()  # the default scores the out-lap
    assert not mask[:, :drop].any()
    np.testing.assert_array_equal(mask.sum(axis=1), np.asarray(ds.n_laps) - drop)
    np.testing.assert_array_equal(mask[:, drop:], base_mask[:, drop:])
    for name in ("h", "w", "y_clean", "y"):
        np.testing.assert_array_equal(np.asarray(getattr(ds, name)), np.asarray(getattr(base, name)))


def test_traffic_adds_a_step_over_consecutive_laps_in_some_stints():
    spec = MISSPEC_VARIANTS["traffic"]
    key = jax.random.key(9)
    base = make_synthetic_dataset(key, 12, noise_s=0.0)
    ds = make_synthetic_dataset(key, 12, noise_s=0.0, misspec=spec)
    traffic = np.asarray(ds.traffic)
    np.testing.assert_allclose(np.asarray(ds.y_clean) - np.asarray(base.y_clean), traffic, atol=1e-4)
    hit = np.flatnonzero(traffic.any(axis=1))
    assert hit.size == round(spec.traffic_stint_fraction * 12)
    low, high = spec.traffic_step_s
    n_laps = np.asarray(ds.n_laps)
    for s in hit:
        laps = np.flatnonzero(traffic[s]) + 1
        np.testing.assert_array_equal(laps, np.arange(laps[0], laps[0] + spec.traffic_laps))
        assert TRAFFIC_MIN_START <= laps[0] and laps[-1] < n_laps[s]
        size = traffic[s, laps - 1]
        assert np.all(size == size[0]) and low <= size[0] <= high


def test_rejects_bad_misspecification():
    with pytest.raises(ValueError, match="noise_df"):
        Misspecification(noise_df=2.0)
    with pytest.raises(ValueError, match="wear_slope_ratio"):
        Misspecification(wear_slope_ratio=0.0)
    with pytest.raises(ValueError, match="drop_first_laps"):
        Misspecification(drop_first_laps=3)
    with pytest.raises(ValueError, match="traffic_step_s"):
        Misspecification(traffic_step_s=(0.5, 0.3))
    with pytest.raises(ValueError, match="traffic_stint_fraction"):
        Misspecification(traffic_stint_fraction=1.5)
    with pytest.raises(ValueError, match="traffic_laps"):
        Misspecification(traffic_laps=0)
    with pytest.raises(ValueError, match="misspec variant"):
        make_synthetic_dataset(jax.random.key(0), 3, misspec="nope")
