"""
Automated Pytest verification suite for the virtual occupancy sensing pipeline.

Tests positive and negative cases, numerical derivative accuracy and convergence
(centred and causal stencils), causality of the deployable features, physical
non-negativity clipping, time-lagged cross-correlation, regression evaluation
metrics, the CO2-threshold engineering baseline, and grouped permutation importance.
"""

import numpy as np
import pytest

from sklearn.linear_model import LinearRegression

from Code.Model_training_and_validation import (
    compute_co2_derivative,
    compute_causal_co2_derivative,
    observed_convergence_order,
    apply_physical_clipping,
    compute_lagged_correlations,
    evaluate_regression_metrics,
    CO2ThresholdRule,
    grouped_permutation_importance,
)


# ==============================================================================
# POSITIVE TEST CASES
# ==============================================================================

def test_positive_sine_wave_derivative_accuracy():
    """
    Positive Test: Verify centred finite-difference derivative against exact
    analytical derivative of C(t) = A * sin(omega * t) + C0 using pytest.approx.
    """
    amplitude = 200.0  # ppm
    baseline = 420.0   # ppm
    period_min = 240.0
    omega = 2.0 * np.pi / period_min
    dt = 0.5           # step size in minutes

    t = np.arange(0.0, period_min, dt)
    c_synthetic = baseline + amplitude * np.sin(omega * t)
    exact_dcdt = amplitude * omega * np.cos(omega * t)

    numerical_dcdt = compute_co2_derivative(c_synthetic, dt_minutes=dt, use_savgol=False)

    # Interior points use O(h^2) centred difference; check interior accuracy
    assert numerical_dcdt[1:-1] == pytest.approx(exact_dcdt[1:-1], rel=1e-3, abs=1e-3), (
        "Interior finite-difference derivative failed to match analytical A*omega*cos(omega*t)."
    )


def test_positive_finite_difference_order_of_accuracy():
    """
    Positive Test: Verify second-order convergence O(h^2) by halving step size h
    and checking observed order p_obs = log(E1/E2) / log(h1/h2) ~ 2.0.
    """
    omega = 2.0 * np.pi / 120.0
    h1 = 2.0
    h2 = 1.0  # Halved step size

    t1 = np.arange(0.0, 120.0, h1)
    t2 = np.arange(0.0, 120.0, h2)

    num_1 = compute_co2_derivative(np.sin(omega * t1), dt_minutes=h1)[1:-1]
    ex_1 = (omega * np.cos(omega * t1))[1:-1]
    err_h1 = np.max(np.abs(num_1 - ex_1))

    num_2 = compute_co2_derivative(np.sin(omega * t2), dt_minutes=h2)[1:-1]
    ex_2 = (omega * np.cos(omega * t2))[1:-1]
    err_h2 = np.max(np.abs(num_2 - ex_2))

    p_obs = np.log(err_h1 / err_h2) / np.log(h1 / h2)

    # Halving h reduces centred-difference truncation error by ~ 2^2 = 4 (p_obs ~ 2.0)
    assert p_obs == pytest.approx(2.0, rel=5e-2), (
        f"Expected second-order convergence p_obs ~ 2.0, got {p_obs:.4f}"
    )


def test_positive_physical_clipping_non_negative():
    """
    Positive Test: Ensure physical clipping maps negative predictions to 0.0
    while preserving positive float values within machine precision.
    """
    raw_preds = np.array([-12.4, -0.001, 0.0, 5.25, 18.9])
    expected = np.array([0.0, 0.0, 0.0, 5.25, 18.9])
    clipped = apply_physical_clipping(raw_preds)

    assert clipped == pytest.approx(expected, rel=1e-12)
    assert np.all(clipped >= 0.0), "Physical clipping allowed negative headcount."


def test_positive_lagged_cross_correlation_peak_detection():
    """
    Positive Test: Verify that a synthetic CO2 signal delayed by +20 minutes
    (2 bins at dt=10 min) produces a peak correlation at tau* = +20 minutes.
    """
    t = np.arange(0, 1000, 10)
    occ = np.sin(2.0 * np.pi * t / 300.0)
    co2_delayed = np.sin(2.0 * np.pi * (t - 20.0) / 300.0)

    lags_min, r_vals = compute_lagged_correlations(occ, co2_delayed, max_lag_minutes=60, dt_minutes=10)
    peak_lag = lags_min[np.argmax(r_vals)]

    assert peak_lag == 20, f"Expected peak lag at +20 min, but found {peak_lag} min."
    assert np.max(r_vals) == pytest.approx(1.0, abs=1e-2)


def test_positive_regression_metrics_calculation():
    """
    Positive Test: Verify MAE, RMSE, R2, and operating-region metrics on a
    known test vector.
    """
    y_true = np.array([0.0, 0.0, 10.0, 20.0])
    y_pred = np.array([-2.0, 2.0, 8.0, 22.0])  # after clipping: [0.0, 2.0, 8.0, 22.0]

    metrics = evaluate_regression_metrics(y_true, y_pred)
    assert metrics["MAE_persons"] == pytest.approx(1.5, rel=1e-9)
    assert metrics["RMSE_persons"] == pytest.approx(np.sqrt(3.0), rel=1e-9)
    assert metrics["MAE_Empty"] == pytest.approx(1.0, rel=1e-9)
    assert metrics["MAE_Occupied"] == pytest.approx(2.0, rel=1e-9)


def test_positive_metrics_all_occupied_returns_nan_empty():
    """
    Positive Test: Verify that MAE_Empty is NaN when no unoccupied samples exist.
    """
    y = np.array([5.0, 10.0, 15.0])
    p = np.array([6.0, 9.0, 14.0])
    m = evaluate_regression_metrics(y, p)
    assert np.isnan(m["MAE_Empty"])
    assert not np.isnan(m["MAE_Occupied"])


def test_positive_savgol_derivative_accuracy():
    """
    Positive Test: Verify Savitzky-Golay derivative accuracy on interior points
    of a smooth synthetic sine wave.
    """
    omega = 2.0 * np.pi / 240.0
    dt = 1.0
    t = np.arange(0.0, 240.0, dt)
    c = 200.0 * np.sin(omega * t)
    d = compute_co2_derivative(c, dt_minutes=dt, use_savgol=True,
                               window_length=7, polyorder=3)
    assert d[5:-5] == pytest.approx((200.0 * omega * np.cos(omega * t))[5:-5],
                                    rel=5e-2, abs=5e-2)


def test_positive_r2_perfect_prediction():
    """
    Positive Test: Verify R2 equals 1.0 for identical observed and predicted vectors.
    """
    y = np.array([0.0, 5.0, 12.0, 20.0])
    m = evaluate_regression_metrics(y, y.copy())
    assert m["R2"] == pytest.approx(1.0)


def test_positive_causal_derivative_second_order_convergence():
    """
    Positive Test: The causal (backward) stencil is also second-order accurate:
    p_obs = log(E1/E2)/log(h1/h2) ~ 2.0 on sin(omega*t) when h is halved twice.
    """
    p_obs = observed_convergence_order("causal", h_values=(2.0, 1.0, 0.5))
    assert p_obs == pytest.approx(2.0, rel=5e-2)


def test_positive_causal_savgol_exact_on_quadratic():
    """
    Positive Test: A trailing Savitzky-Golay filter with polyorder=2 differentiates a
    quadratic C(t) = 3 + 2t + 0.05 t^2 exactly once a full window of history exists.
    """
    t = np.arange(0.0, 100.0, 10.0)
    c = 3.0 + 2.0 * t + 0.05 * t ** 2
    d = compute_causal_co2_derivative(c, dt_minutes=10.0, use_savgol=True,
                                      window_length=5, polyorder=2)
    assert d[4:] == pytest.approx((2.0 + 0.1 * t)[4:], rel=1e-9)


def test_positive_causal_derivative_ignores_future_samples():
    """
    Positive Test (leakage guard): changing samples at index >= k must not change
    the causal derivative at indices < k, for both stencil variants. The centred
    Savitzky-Golay filter fails this property, which is why it cannot be a deployed feature.
    """
    rng = np.random.default_rng(0)
    c = rng.normal(500.0, 50.0, 60)
    c_future_changed = c.copy()
    c_future_changed[40:] += 1000.0
    for use_savgol in (False, True):
        a = compute_causal_co2_derivative(c, 10.0, use_savgol=use_savgol)
        b = compute_causal_co2_derivative(c_future_changed, 10.0, use_savgol=use_savgol)
        assert np.array_equal(a[:40], b[:40])
    leaky_a = compute_co2_derivative(c, 10.0, use_savgol=True)
    leaky_b = compute_co2_derivative(c_future_changed, 10.0, use_savgol=True)
    assert not np.array_equal(leaky_a[:40], leaky_b[:40])


def test_positive_threshold_rule_known_values():
    """
    Positive Test: With threshold 600 ppm the rule predicts 0 below it and the mean
    training headcount of the above-threshold samples (15 persons) above it.
    """
    x = np.array([[400.0], [500.0], [700.0], [900.0]])
    y = np.array([0.0, 0.0, 10.0, 20.0])
    rule = CO2ThresholdRule(threshold_ppm=600.0).fit(x, y)
    assert rule.occupied_level_ == pytest.approx(15.0)
    assert rule.predict(x) == pytest.approx([0.0, 0.0, 15.0, 15.0])


def test_positive_tuned_threshold_recovers_step_change():
    """
    Positive Test: On synthetic data with an exact step (450 ppm empty / 800 ppm
    occupied with 20 persons), the tuned threshold lies between the two levels and the
    training MAE is zero.
    """
    x = np.array([450.0] * 30 + [800.0] * 30)
    y = np.array([0.0] * 30 + [20.0] * 30)
    rule = CO2ThresholdRule(tune=True).fit(x, y)
    assert 450.0 <= rule.threshold_ppm_ < 800.0
    assert rule.predict(x) == pytest.approx(y, abs=1e-12)


def test_positive_grouped_permutation_importance_identifies_driver():
    """
    Positive Test: When y depends only on x0, shuffling the group {x0} raises RMSE
    substantially while shuffling the irrelevant group {x1} changes it by ~0.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = 5.0 * X[:, 0] + 40.0          # offset keeps y > 0 so clipping never alters predictions
    model = LinearRegression().fit(X, y)
    imp, base = grouped_permutation_importance(
        model, X, y, ["x0", "x1"], {"driver": ["x0"], "noise": ["x1"]},
        n_repeats=10, random_state=0)
    table = imp.set_index("Feature_Group")["RMSE_increase_mean"]
    assert base == pytest.approx(0.0, abs=1e-8)
    assert table["driver"] > 1.0
    assert table["noise"] == pytest.approx(0.0, abs=1e-8)


# ==============================================================================
# NEGATIVE TEST CASES
# ==============================================================================

def test_negative_derivative_non_positive_step_size():
    """
    Negative Test: Verify compute_co2_derivative raises ValueError when dt <= 0.
    """
    valid_co2 = np.array([400.0, 420.0, 450.0, 430.0])
    with pytest.raises(ValueError):
        compute_co2_derivative(valid_co2, dt_minutes=0.0)
    with pytest.raises(ValueError):
        compute_co2_derivative(valid_co2, dt_minutes=-10.0)


def test_negative_derivative_corrupted_nan_or_inf_input():
    """
    Negative Test: Verify compute_co2_derivative safely rejects NaN or Inf values.
    """
    corrupted_nan = np.array([410.0, np.nan, 430.0, 440.0])
    corrupted_inf = np.array([410.0, np.inf, 430.0, 440.0])

    with pytest.raises(ValueError):
        compute_co2_derivative(corrupted_nan, dt_minutes=10.0)
    with pytest.raises(ValueError):
        compute_co2_derivative(corrupted_inf, dt_minutes=10.0)


def test_negative_derivative_insufficient_samples():
    """
    Negative Test: Verify compute_co2_derivative raises ValueError if fewer
    than 3 samples are provided.
    """
    with pytest.raises(ValueError):
        compute_co2_derivative(np.array([400.0, 410.0]), dt_minutes=10.0)


def test_negative_clipping_empty_or_nan():
    """
    Negative Test: Verify apply_physical_clipping raises ValueError on empty
    or NaN arrays.
    """
    with pytest.raises(ValueError):
        apply_physical_clipping(np.array([]))
    with pytest.raises(ValueError):
        apply_physical_clipping(np.array([1.0, np.nan, 3.0]))


def test_negative_lagged_correlation_mismatched_lengths():
    """
    Negative Test: Verify compute_lagged_correlations raises ValueError when
    occupancy and CO2 arrays have different lengths.
    """
    occ = np.zeros(50)
    co2 = np.zeros(45)
    with pytest.raises(ValueError):
        compute_lagged_correlations(occ, co2, max_lag_minutes=20, dt_minutes=10)


def test_negative_lagged_correlation_series_too_short():
    """
    Negative Test: Verify compute_lagged_correlations raises ValueError when
    time series length is insufficient for the requested maximum lag.
    """
    occ = np.zeros(15)
    co2 = np.zeros(15)
    with pytest.raises(ValueError):
        compute_lagged_correlations(occ, co2, max_lag_minutes=120, dt_minutes=10)


def test_negative_savgol_invalid_window():
    """
    Negative Test: Verify compute_co2_derivative raises ValueError for even
    window_length or window_length <= polyorder.
    """
    c = np.sin(np.linspace(0, 1, 50))
    with pytest.raises(ValueError):
        compute_co2_derivative(c, dt_minutes=1.0, use_savgol=True,
                               window_length=4, polyorder=2)
    with pytest.raises(ValueError):
        compute_co2_derivative(c, dt_minutes=1.0, use_savgol=True,
                               window_length=3, polyorder=5)


def test_negative_causal_derivative_invalid_inputs():
    """
    Negative Test: compute_causal_co2_derivative rejects dt <= 0, NaN/Inf, fewer than
    3 samples, and invalid Savitzky-Golay settings.
    """
    ok = np.array([400.0, 410.0, 430.0, 450.0, 440.0, 435.0])
    with pytest.raises(ValueError):
        compute_causal_co2_derivative(ok, dt_minutes=0.0)
    with pytest.raises(ValueError):
        compute_causal_co2_derivative(np.array([400.0, np.nan, 420.0]), dt_minutes=10.0)
    with pytest.raises(ValueError):
        compute_causal_co2_derivative(np.array([400.0, 410.0]), dt_minutes=10.0)
    with pytest.raises(ValueError):
        compute_causal_co2_derivative(ok, dt_minutes=10.0, use_savgol=True,
                                      window_length=4, polyorder=2)


def test_negative_threshold_rule_rejects_nan_and_multicolumn_input():
    """
    Negative Test: CO2ThresholdRule raises ValueError on NaN CO2, on more than one
    input column, and on a target whose length does not match.
    """
    with pytest.raises(ValueError):
        CO2ThresholdRule().fit(np.array([[500.0], [np.nan]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        CO2ThresholdRule().fit(np.ones((5, 2)), np.zeros(5))
    with pytest.raises(ValueError):
        CO2ThresholdRule().fit(np.array([[500.0], [600.0]]), np.array([0.0]))


def test_negative_grouped_permutation_importance_invalid_arguments():
    """
    Negative Test: grouped_permutation_importance raises ValueError for an unknown
    feature name, an empty group, and inconsistent shapes.
    """
    X = np.random.default_rng(0).normal(size=(20, 2))
    y = X[:, 0]
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError):
        grouped_permutation_importance(model, X, y, ["a", "b"], {"g": ["not_a_feature"]})
    with pytest.raises(ValueError):
        grouped_permutation_importance(model, X, y, ["a", "b"], {"g": []})
    with pytest.raises(ValueError):
        grouped_permutation_importance(model, X, y[:-1], ["a", "b"], {"g": ["a"]})