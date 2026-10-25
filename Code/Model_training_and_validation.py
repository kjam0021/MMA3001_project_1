"""
Machine learning pipeline for virtual occupancy sensing using CO2 dynamics.

This module implements chronological data splitting, time-lagged cross-correlation,
finite-difference derivative estimation (centred for offline analysis, causal for
deployable features), regression model training, an engineering CO2-threshold
baseline, cross-validation, feature-influence and time-feature ablation studies,
residual diagnostics, and out-of-regime failure mode analysis.
"""

import os
import re
import platform
import timeit
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_coeffs, savgol_filter

import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.set_option("display.precision", 4)


# ==============================================================================
# 1. NUMERICAL DIFFERENTIATION & CROSS-CORRELATION
# ==============================================================================

def compute_co2_derivative(co2_series, dt_minutes=10.0, use_savgol=False,
                           window_length=5, polyorder=2):
    """
    Estimate the physical time derivative dC/dt of CO2 concentration.

    Differentiates with respect to physical time spacing `dt_minutes` rather than
    sample index. Interior points use the second-order centred finite-difference
    stencil O(h^2), while boundaries use first-order forward/backward stencils O(h).

    Parameters
    ----------
    co2_series : array_like
        1D array or Series of CO2 concentrations in ppm.
    dt_minutes : float, default=10.0
        Physical sampling interval in minutes. Must be strictly positive.
    use_savgol : bool, default=False
        If True, applies a Savitzky-Golay smoothing filter to suppress sensor
        measurement noise before differentiation.
    window_length : int, default=5
        Window length for the Savitzky-Golay filter (must be odd and >= polyorder + 2).
    polyorder : int, default=2
        Polynomial degree for the Savitzky-Golay local fit.

    Returns
    -------
    dcdt : numpy.ndarray
        Estimated rate of change dC/dt in units of ppm/min.

    Raises
    ------
    TypeError
        If `co2_series` cannot be converted to a numeric float64 array.
    ValueError
        If `co2_series` has fewer than 3 samples, contains NaNs/Infs, or if
        `dt_minutes` <= 0.
    """
    arr = np.asarray(co2_series, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError("co2_series must be a 1D array with at least 3 samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("co2_series contains NaN or infinite values.")
    if dt_minutes <= 0:
        raise ValueError("Physical step size dt_minutes must be strictly positive.")

    if use_savgol:
        if window_length % 2 == 0 or window_length > arr.size or window_length <= polyorder:
            raise ValueError("Invalid window_length or polyorder for Savitzky-Golay filter.")
        return savgol_filter(arr, window_length=window_length, polyorder=polyorder,
                             deriv=1, delta=float(dt_minutes))

    return np.gradient(arr, float(dt_minutes))


def compute_causal_co2_derivative(co2_series, dt_minutes=10.0, use_savgol=False,
                                  window_length=5, polyorder=2):
    """
    Estimate dC/dt using ONLY present and past samples (deployable in real time).

    The centred stencil in `compute_co2_derivative` uses C(t + dt), i.e. a future
    reading. Because CO2 lags occupancy by ~20 min, a feature built from future CO2
    leaks information about the present headcount and cannot exist at deployment.
    This function uses backward (trailing) stencils only:

    - i >= 2 : second-order backward difference
      ``(3 C_i - 4 C_{i-1} + C_{i-2}) / (2 dt)``, truncation error O(dt^2);
    - i == 1 : first-order backward difference, O(dt);
    - i == 0 : no history is available, so the derivative is set to 0.0 ppm/min.

    With ``use_savgol=True`` the trailing Savitzky-Golay derivative (the fitted
    polynomial is evaluated at the NEWEST sample of the window) replaces the
    backward-difference value for every index ``i >= window_length - 1``.

    Parameters
    ----------
    co2_series : array_like
        1D array of CO2 concentrations in ppm (at least 3 finite samples).
    dt_minutes : float, default=10.0
        Physical sampling interval in minutes; must be strictly positive.
    use_savgol : bool, default=False
        If True, use the trailing Savitzky-Golay derivative where a full window exists.
    window_length : int, default=5
        Odd Savitzky-Golay window length (> polyorder, <= number of samples).
    polyorder : int, default=2
        Polynomial order of the local least-squares fit.

    Returns
    -------
    dcdt : numpy.ndarray
        Causal rate of change in ppm/min; ``dcdt[i]`` depends only on ``co2[:i+1]``.

    Raises
    ------
    ValueError
        If the input is not 1D, has fewer than 3 samples, contains NaN/Inf, if
        ``dt_minutes <= 0``, or if the Savitzky-Golay settings are invalid.
    """
    arr = np.asarray(co2_series, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError("co2_series must be a 1D array with at least 3 samples.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("co2_series contains NaN or infinite values.")
    if dt_minutes <= 0:
        raise ValueError("Physical step size dt_minutes must be strictly positive.")
    dt = float(dt_minutes)

    out = np.zeros_like(arr)
    out[1] = (arr[1] - arr[0]) / dt
    out[2:] = (3.0 * arr[2:] - 4.0 * arr[1:-1] + arr[:-2]) / (2.0 * dt)

    if use_savgol:
        if window_length % 2 == 0 or window_length > arr.size or window_length <= polyorder:
            raise ValueError("Invalid window_length or polyorder for Savitzky-Golay filter.")
        # pos = window_length - 1 evaluates the local polynomial at the newest sample.
        coeffs = savgol_coeffs(window_length, polyorder, deriv=1, delta=dt,
                               pos=window_length - 1, use="dot")
        out[window_length - 1:] = sliding_window_view(arr, window_length) @ coeffs
    return out


def observed_convergence_order(method="centred", h_values=(2.0, 1.0, 0.5), period_min=120.0):
    """
    Measure the observed order of accuracy p_obs of a derivative stencil.

    Differentiates ``sin(omega t)`` (exact derivative ``omega cos(omega t)``) on
    successively halved step sizes and returns
    ``p_obs = log(E_k / E_{k+1}) / log(h_k / h_{k+1})`` from the maximum interior
    errors. A second-order scheme gives p_obs ~ 2.

    Parameters
    ----------
    method : {"centred", "causal"}
        Stencil to verify (`compute_co2_derivative` or `compute_causal_co2_derivative`).
    h_values : sequence of float
        Strictly decreasing step sizes in minutes (at least two).
    period_min : float
        Period of the test sinusoid in minutes.

    Returns
    -------
    p_obs : numpy.ndarray
        Observed order between each consecutive pair of step sizes.

    Raises
    ------
    ValueError
        If ``method`` is unknown or fewer than two step sizes are given.
    """
    if method not in ("centred", "causal"):
        raise ValueError("method must be 'centred' or 'causal'.")
    if len(h_values) < 2:
        raise ValueError("At least two step sizes are required.")
    omega = 2.0 * np.pi / period_min
    errs = []
    for h in h_values:
        t = np.arange(0.0, period_min, h)
        exact = omega * np.cos(omega * t)
        if method == "centred":
            num = compute_co2_derivative(np.sin(omega * t), dt_minutes=h)
            errs.append(np.max(np.abs(num[1:-1] - exact[1:-1])))
        else:
            num = compute_causal_co2_derivative(np.sin(omega * t), dt_minutes=h)
            errs.append(np.max(np.abs(num[2:] - exact[2:])))
    h_arr = np.asarray(h_values, dtype=np.float64)
    e_arr = np.asarray(errs)
    return np.log(e_arr[:-1] / e_arr[1:]) / np.log(h_arr[:-1] / h_arr[1:])


def apply_physical_clipping(predictions):
    """
    Enforce physical non-negativity constraint on predicted headcount.

    In accordance with physical conservation laws, building occupancy cannot
    be negative: N_hat(t) = max(0, y_pred(t)).

    Parameters
    ----------
    predictions : array_like
        Raw model predictions of headcount (persons).

    Returns
    -------
    clipped : numpy.ndarray
        Non-negative predictions in persons.

    Raises
    ------
    ValueError
        If input array is empty or contains NaN values.
    """
    arr = np.asarray(predictions, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Predictions array cannot be empty.")
    if np.any(np.isnan(arr)):
        raise ValueError("Predictions array cannot contain NaN values.")
    return np.maximum(0.0, arr)


def compute_lagged_correlations(occ_series, co2_series, max_lag_minutes=120, dt_minutes=10):
    """
    Compute Pearson cross-correlation R(tau) across time lags tau.

    Evaluates correlation between occupancy N(t) and shifted CO2 C(t + tau)
    for tau in [-max_lag_minutes, +max_lag_minutes]. Positive tau means CO2
    lags behind occupancy (physical gas dispersion delay).

    Parameters
    ----------
    occ_series : array_like
        1D array of ground-truth occupancy headcount (persons).
    co2_series : array_like
        1D array of CO2 measurements (ppm) or derivatives (ppm/min).
    max_lag_minutes : int, default=120
        Maximum lag in minutes in both negative and positive directions.
    dt_minutes : int, default=10
        Step size in minutes per bin.

    Returns
    -------
    lags_min : numpy.ndarray
        Array of lag values in minutes.
    pearson_r : numpy.ndarray
        Pearson correlation coefficients at each lag.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, fewer than 10 samples, or invalid lag.
    """
    occ = np.asarray(occ_series, dtype=np.float64)
    co2 = np.asarray(co2_series, dtype=np.float64)
    if occ.shape != co2.shape or occ.ndim != 1:
        raise ValueError("occ_series and co2_series must be 1D arrays of equal length.")
    max_bins = int(max_lag_minutes // dt_minutes)
    if max_bins <= 0 or occ.size <= 2 * max_bins + 2:
        raise ValueError("Time series too short for the requested max_lag_minutes.")

    lags_bins = np.arange(-max_bins, max_bins + 1)
    lags_min = lags_bins * dt_minutes
    pearson_r = np.zeros(lags_bins.size, dtype=np.float64)

    for idx, k in enumerate(lags_bins):
        if k < 0:
            x_slice = occ[-k:]
            y_slice = co2[:k]
        elif k > 0:
            x_slice = occ[:-k]
            y_slice = co2[k:]
        else:
            x_slice = occ
            y_slice = co2

        valid = np.isfinite(x_slice) & np.isfinite(y_slice)
        if np.sum(valid) > 2 and np.std(x_slice[valid]) > 0 and np.std(y_slice[valid]) > 0:
            pearson_r[idx] = np.corrcoef(x_slice[valid], y_slice[valid])[0, 1]
        else:
            pearson_r[idx] = 0.0

    return lags_min, pearson_r


# ==============================================================================
# 1b. ENGINEERING BASELINE & FEATURE-INFLUENCE UTILITIES
# ==============================================================================

class CO2ThresholdRule(BaseEstimator, RegressorMixin):
    """
    Engineering baseline: the CO2-threshold occupancy heuristic used in practice.

    Rule: if ``C > threshold_ppm`` the zone is "occupied" and the predicted headcount
    is the mean training headcount over samples above the threshold; otherwise the
    prediction is 0 persons. With ``tune=True`` the threshold is chosen on the
    TRAINING data only, by grid search minimising training MAE.

    Unlike ``DummyRegressor`` (a statistical floor), this is the credible
    non-ML alternative that a building engineer would deploy, so beating it is the
    evidence that a learned model adds value.

    Parameters
    ----------
    threshold_ppm : float, default=600.0
        Occupied/unoccupied CO2 threshold in ppm (used when ``tune`` is False).
    tune : bool, default=False
        If True, search ``[grid_min_ppm, grid_max_ppm]`` for the best threshold.
    grid_min_ppm, grid_max_ppm, grid_step_ppm : float
        Search grid for the tuned threshold in ppm.

    Attributes
    ----------
    threshold_ppm_ : float
        Threshold actually used after fitting.
    occupied_level_ : float
        Predicted headcount (persons) when ``C > threshold_ppm_``.
    """

    def __init__(self, threshold_ppm=600.0, tune=False,
                 grid_min_ppm=420.0, grid_max_ppm=1200.0, grid_step_ppm=10.0):
        self.threshold_ppm = threshold_ppm
        self.tune = tune
        self.grid_min_ppm = grid_min_ppm
        self.grid_max_ppm = grid_max_ppm
        self.grid_step_ppm = grid_step_ppm

    @staticmethod
    def _as_co2(X):
        """Return CO2 as a finite 1D float array from a 1D array or single-column 2D array."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("CO2ThresholdRule expects a non-empty 1D array or a single CO2 column.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("CO2 input contains NaN or infinite values.")
        return arr

    def fit(self, X, y):
        """Fit the occupied-level (and the threshold if ``tune``) on training data."""
        co2 = self._as_co2(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.shape != co2.shape or not np.all(np.isfinite(y_arr)):
            raise ValueError("y must be finite and match the number of CO2 samples.")
        if self.tune:
            if self.grid_step_ppm <= 0 or self.grid_max_ppm < self.grid_min_ppm:
                raise ValueError("Invalid threshold search grid.")
            grid = np.arange(self.grid_min_ppm, self.grid_max_ppm + self.grid_step_ppm, self.grid_step_ppm)
        else:
            grid = np.array([float(self.threshold_ppm)])

        best = None
        for thr in grid:
            above = co2 > thr
            level = float(y_arr[above].mean()) if np.any(above) else 0.0
            mae = float(np.mean(np.abs(y_arr - np.where(above, level, 0.0))))
            if best is None or mae < best[0]:
                best = (mae, float(thr), level)
        _, self.threshold_ppm_, self.occupied_level_ = best
        return self

    def predict(self, X):
        """Predict headcount in persons: ``occupied_level_`` above the threshold, else 0."""
        co2 = self._as_co2(X)
        return np.where(co2 > self.threshold_ppm_, self.occupied_level_, 0.0)


def grouped_permutation_importance(model, X, y, feature_names, groups, n_repeats=20, random_state=0):
    """
    Grouped permutation importance measured as the increase in RMSE.

    Rows of every column in a group are shuffled TOGETHER (same permutation), so
    correlated features such as ``co2_active, co2_lag1, co2_lag2`` or ``sin_time,
    cos_time`` are assessed as one physical quantity instead of splitting (and
    hiding) their importance. Evaluate on held-out data, never on training data.

    Parameters
    ----------
    model : fitted estimator with ``predict``
        Fitted pipeline; predictions are physically clipped at 0 persons.
    X : array_like, shape (n_samples, n_features)
        Held-out feature matrix, columns ordered as ``feature_names``.
    y : array_like, shape (n_samples,)
        Observed headcount in persons.
    feature_names : sequence of str
        Column names of ``X``.
    groups : dict[str, list[str]]
        Mapping of group label to the feature names that are permuted together.
    n_repeats : int, default=20
        Number of random permutations per group.
    random_state : int, default=0
        Seed for reproducibility.

    Returns
    -------
    importance_df : pandas.DataFrame
        Columns ``Feature_Group``, ``Features``, ``RMSE_increase_mean`` and
        ``RMSE_increase_std`` (persons), sorted by decreasing importance.
    base_rmse : float
        RMSE (persons) of the unpermuted model on ``X``.

    Raises
    ------
    ValueError
        If shapes are inconsistent, a group is empty, or a feature name is unknown.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    names = list(feature_names)
    if X_arr.ndim != 2 or X_arr.shape[1] != len(names) or X_arr.shape[0] != y_arr.size:
        raise ValueError("X, y and feature_names have inconsistent shapes.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")

    def _rmse(Xm):
        pred = apply_physical_clipping(model.predict(Xm))
        return float(np.sqrt(mean_squared_error(y_arr, pred)))

    base = _rmse(X_arr)
    rng = np.random.default_rng(random_state)
    rows = []
    for label, cols in groups.items():
        if len(cols) == 0:
            raise ValueError(f"Feature group '{label}' is empty.")
        idx = [names.index(c) for c in cols]  # list.index raises ValueError if unknown
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(X_arr.shape[0])
            X_p = X_arr.copy()
            X_p[:, idx] = X_arr[perm][:, idx]
            deltas.append(_rmse(X_p) - base)
        rows.append({"Feature_Group": label, "Features": ", ".join(cols),
                     "RMSE_increase_mean": float(np.mean(deltas)),
                     "RMSE_increase_std": float(np.std(deltas))})
    out = pd.DataFrame(rows).sort_values("RMSE_increase_mean", ascending=False).reset_index(drop=True)
    return out, base


def df_to_markdown(df, floatfmt="{:.3f}"):
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==============================================================================
# 2. DATA SPLITTING & FEATURE ENGINEERING
# ==============================================================================

def load_and_split_sensor_869(csv_path, causal_features=True):
    """
    Filter dataset to active co2_sensor_869 window and split chronologically.

    Uses a strict time-ordered split at Melbourne midnight (00:00:00) to prevent
    temporal data leakage across Train (70%), Validation (15%), and Withheld
    Test (15%) sets.

    Parameters
    ----------
    csv_path : str
        Path to `synchronized_co2_occupancy_10min.csv`.
    causal_features : bool, default=True
        If True (deployable), ``dcdt_raw`` / ``dcdt_savgol`` use trailing stencils that
        depend only on present and past CO2. If False, the centred/Savitzky-Golay
        stencils are used, which read future samples (offline analysis only; kept for
        a leakage sensitivity check).

    Returns
    -------
    train_df : pandas.DataFrame
        Training set (2024-02-29 to 2024-05-12, 74 calendar days).
    val_df : pandas.DataFrame
        Validation set (2024-05-13 to 2024-05-28, 16 calendar days).
    test_df : pandas.DataFrame
        Withheld final test set (2024-05-29 to 2024-06-14, 17 calendar days).
    sensor_326_df : pandas.DataFrame
        Out-of-regime dataset from co2_sensor_326 (2025-2026) for failure analysis.
    """
    df = pd.read_csv(csv_path)
    time_col = "timestamp_melbourne" if "timestamp_melbourne" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert("Australia/Melbourne")
    df = df.sort_values(time_col).reset_index(drop=True)
    df["timestamp_melbourne"] = df[time_col]

    # Filter active primary sensor window: co2_sensor_869
    if "co2_sensor_869" in df.columns:
        df_869 = df[df["co2_sensor_869"].notna()].copy()
        df_869["co2_active"] = df_869["co2_sensor_869"].astype(float)
    else:
        df_869 = df[(df[time_col] >= "2024-02-29") & (df[time_col] < "2024-06-15")].copy()
        df_869["co2_active"] = df_869["co2_ppm"].astype(float)

    # Filter out-of-regime secondary sensor window: co2_sensor_326
    if "co2_sensor_326" in df.columns:
        df_326 = df[df["co2_sensor_326"].notna()].copy()
        df_326["co2_active"] = df_326["co2_sensor_326"].astype(float)
    else:
        df_326 = df[df[time_col] >= "2025-01-01"].copy()
        df_326["co2_active"] = df_326["co2_ppm"].astype(float)

    # Engineer physical & diurnal features (respecting causal boundaries)
    for sub_df in (df_869, df_326):
        if len(sub_df) < 5:
            continue
        deriv_fn = compute_causal_co2_derivative if causal_features else compute_co2_derivative
        sub_df["dcdt_raw"] = deriv_fn(
            sub_df["co2_active"].values, dt_minutes=10.0, use_savgol=False
        )
        sub_df["dcdt_savgol"] = deriv_fn(
            sub_df["co2_active"].values, dt_minutes=10.0, use_savgol=True
        )
        sub_df["co2_lag1"] = sub_df["co2_active"].shift(1).bfill()
        sub_df["co2_lag2"] = sub_df["co2_active"].shift(2).bfill()

        hour_float = sub_df[time_col].dt.hour + sub_df[time_col].dt.minute / 60.0
        sub_df["sin_time"] = np.sin(2.0 * np.pi * hour_float / 24.0)
        sub_df["cos_time"] = np.cos(2.0 * np.pi * hour_float / 24.0)

        if "temp_c" not in sub_df.columns:
            sub_df["temp_c"] = 21.0
        if "humidity_rh" not in sub_df.columns:
            sub_df["humidity_rh"] = 50.0

    # Strict chronological split at Melbourne midnight
    cal_dates = df_869[time_col].dt.date
    train_end = pd.to_datetime("2024-05-12").date()
    val_end = pd.to_datetime("2024-05-28").date()

    train_df = df_869[cal_dates <= train_end].copy()
    val_df = df_869[(cal_dates > train_end) & (cal_dates <= val_end)].copy()
    test_df = df_869[cal_dates > val_end].copy()

    return train_df, val_df, test_df, df_326


# ==============================================================================
# 3. REGRESSION METRICS & OPERATING-REGION DIAGNOSTICS
# ==============================================================================

def evaluate_regression_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics and operating-region checks.

    Computes global MAE, RMSE, and R2, plus localized MAE across distinct
    operating regions (Unoccupied N=0 vs Occupied N>0).

    Parameters
    ----------
    y_true : array_like
        Ground-truth occupancy headcount in persons.
    y_pred : array_like
        Predicted occupancy headcount in persons.

    Returns
    -------
    metrics : dict
        Dictionary containing MAE_persons, RMSE_persons, R2,
        MAE_Empty, and MAE_Occupied.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = apply_physical_clipping(y_pred)

    mse = mean_squared_error(y_t, y_p)
    mae_all = mean_absolute_error(y_t, y_p)
    rmse_all = np.sqrt(mse)
    r2_all = r2_score(y_t, y_p)

    empty_mask = (y_t == 0)
    occ_mask = (y_t > 0)

    mae_empty = mean_absolute_error(y_t[empty_mask], y_p[empty_mask]) if np.any(empty_mask) else np.nan
    mae_occ = mean_absolute_error(y_t[occ_mask], y_p[occ_mask]) if np.any(occ_mask) else np.nan

    return {
        "MAE_persons": mae_all,
        "RMSE_persons": rmse_all,
        "R2": r2_all,
        "MAE_Empty": mae_empty,
        "MAE_Occupied": mae_occ,
    }


# Keep alias for backwards compatibility with existing imports
evaluate_mma3001_regression = evaluate_regression_metrics


# ==============================================================================
# 4. EMPIRICAL TIMING & ARITHMETIC INTENSITY BENCHMARK
# ==============================================================================

def benchmark_numerical_performance(co2_array, n_runs=20):
    """
    Benchmark execution time and estimate FLOPS and Arithmetic Intensity.

    Compares a Python loop centred-difference implementation against
    vectorized `numpy.gradient` using `timeit.repeat` averaged across multiple
    runs to account for background operating system variation.

    Parameters
    ----------
    co2_array : numpy.ndarray
        1D array of CO2 readings (float64, 8 bytes per element).
    n_runs : int, default=20
        Number of repetitions for statistical timing confidence.

    Returns
    -------
    perf_summary : dict
        Timing statistics (mean/std in ms), speedup factor, FLOPS, and
        Arithmetic Intensity (FLOPS/Byte).
    """
    arr = np.asarray(co2_array, dtype=np.float64)
    n_pts = arr.size
    dt = 10.0

    def py_loop_diff():
        out = np.empty_like(arr)
        out[0] = (arr[1] - arr[0]) / dt
        for i in range(1, n_pts - 1):
            out[i] = (arr[i + 1] - arr[i - 1]) / (2.0 * dt)
        out[-1] = (arr[-1] - arr[-2]) / dt
        return out

    def np_vec_diff():
        return np.gradient(arr, dt)

    loop_times = timeit.repeat(py_loop_diff, repeat=5, number=n_runs)
    vec_times = timeit.repeat(np_vec_diff, repeat=5, number=n_runs)

    loop_ms = (np.mean(loop_times) / n_runs) * 1000.0
    loop_std_ms = (np.std(loop_times) / n_runs) * 1000.0
    vec_ms = (np.mean(vec_times) / n_runs) * 1000.0
    vec_std_ms = (np.std(vec_times) / n_runs) * 1000.0

    # Each interior point performs 1 subtraction + 1 division = 2 FLOPs
    # Memory traffic: 1 float64 read + 1 float64 write streamed per element (16 bytes/pt)
    total_flops = 2.0 * n_pts
    total_bytes = 2.0 * 8.0 * n_pts
    arithmetic_intensity = total_flops / total_bytes
    achieved_mflops = (total_flops / (vec_ms / 1000.0)) / 1e6

    return {
        "N_samples": n_pts,
        "Python_Loop_ms": loop_ms,
        "Python_Loop_std_ms": loop_std_ms,
        "NumPy_Vectorized_ms": vec_ms,
        "NumPy_Vectorized_std_ms": vec_std_ms,
        "Speedup_Factor": loop_ms / max(vec_ms, 1e-9),
        "Arithmetic_Intensity_FLOP_per_Byte": arithmetic_intensity,
        "Vectorized_MFLOPS_per_sec": achieved_mflops,
    }


# ==============================================================================
# 5. PLOTTING HELPER FUNCTIONS
# ==============================================================================

def _slugify(text):
    """Convert a model name into a clean filename slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug


def save_individual_model_plot(model_name, timestamps, y_true, y_pred, metrics_dict, output_dir):
    """
    Save a 3-panel diagnostic figure for a single regression method.

    Panel 1: Predicted vs Observed scatter with 1:1 identity line.
    Panel 2: Residuals (Observed - Predicted) vs Predicted values.
    Panel 3: Time-series tracking over a representative 5-day window of the test set.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    residuals = y_t - y_p

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax_parity = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    # Panel 1: Predicted vs Observed
    max_val = max(float(np.max(y_t)), float(np.max(y_p)), 1.0)
    ax_parity.scatter(y_t, y_p, alpha=0.35, s=18, color="tab:blue", label="Test samples")
    ax_parity.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Ideal 1:1 line")
    ax_parity.set_xlabel("Observed Headcount [persons]")
    ax_parity.set_ylabel("Predicted Headcount [persons]")
    ax_parity.set_title(
        f"Predicted vs Observed\n"
        f"(MAE={metrics_dict['MAE_persons']:.2f}, RMSE={metrics_dict['RMSE_persons']:.2f}, R$^2$={metrics_dict['R2']:.3f})"
    )
    ax_parity.legend(loc="upper left")
    ax_parity.grid(True, alpha=0.3)

    # Panel 2: Residual vs Predicted
    ax_resid.scatter(y_p, residuals, alpha=0.35, s=18, color="tab:orange")
    ax_resid.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax_resid.set_xlabel("Predicted Headcount [persons]")
    ax_resid.set_ylabel(r"Residual $e = y - \hat{y}$ [persons]")
    ax_resid.set_title(
        f"Residuals vs Predicted\n"
        f"(Empty MAE={metrics_dict['MAE_Empty']:.2f}, Occupied MAE={metrics_dict['MAE_Occupied']:.2f})"
    )
    ax_resid.grid(True, alpha=0.3)

    # Panel 3: Time-Series Window (first 5 days = 720 bins of 10-min intervals)
    n_window = min(len(y_t), 720)
    t_win = timestamps.iloc[:n_window]
    ax_ts.plot(t_win, y_t[:n_window], color="black", linewidth=1.8, label="Observed Headcount")
    ax_ts.plot(t_win, y_p[:n_window], color="tab:red", linestyle="--", linewidth=1.6, label=f"Predicted ({model_name})")
    ax_ts.fill_between(t_win, y_t[:n_window], y_p[:n_window], color="tab:red", alpha=0.15, label="Prediction Error")
    ax_ts.set_xlabel("Melbourne Local Time")
    ax_ts.set_ylabel("Headcount [persons]")
    ax_ts.set_title("Test Set Time-Series Tracking (5-Day Window)")
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    fig.suptitle(f"Regression Diagnostic Report: {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = f"individual_{_slugify(model_name)}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close(fig)


def save_combined_comparison_plots(test_df, target_col, trained_pipelines, results_df, cv_fold_records, output_dir):
    """
    Save combined multi-model comparison plots onto shared figures.

    Generates:
    1. `fig2_all_models_timeseries_overlay.png`: All regression methods plotted on
       a single time-series figure against observed occupancy.
    2. `fig3_all_models_parity_grid.png`: Multi-panel subplot grid of Predicted vs
       Observed for all regression methods on shared axes.
    3. `fig4_model_metrics_comparison_bars.png`: Bar chart comparing Validation and
       Test MAE, RMSE, and R2 across all regression methods.
    4. `fig5_cross_validation_fold_boxplots.png`: Boxplot of 5-fold cross-validation
       RMSE across all regression methods.
    """
    timestamps = test_df["timestamp_melbourne"]
    y_test = test_df[target_col].values

    # ------------------------------------------------------------------
    # Plot A: All regression methods overlaid on one time-series plot
    # ------------------------------------------------------------------
    n_win = min(len(y_test), 576)  # 4-day detailed window for visual clarity
    t_win = timestamps.iloc[:n_win]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_win, y_test[:n_win], color="black", linewidth=2.4, label="Observed Headcount (Ground Truth)", zorder=5)

    cmap = plt.get_cmap("tab10")
    for idx, (name, (_, _, _, test_pred)) in enumerate(trained_pipelines.items()):
        line_style = ":" if "Baseline" in name else "-"
        alpha_val = 0.65 if "Baseline" in name else 0.85
        ax.plot(
            t_win,
            test_pred[:n_win],
            linestyle=line_style,
            linewidth=1.5,
            color=cmap(idx % 10),
            alpha=alpha_val,
            label=name,
        )

    ax.set_xlabel("Melbourne Local Time")
    ax.set_ylabel("Occupancy Headcount [persons]")
    ax.set_title("Comparison of All Regression Methods on Test Set Time Series")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_all_models_timeseries_overlay.png"), dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot B: All models Predicted vs Observed on one multi-panel grid
    # ------------------------------------------------------------------
    n_models = len(trained_pipelines)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    max_lim = max(float(np.max(y_test)), 1.0)
    for idx, (name, (_, _, _, test_pred)) in enumerate(trained_pipelines.items()):
        ax_i = axes_flat[idx]
        r2_val = r2_score(y_test, test_pred)
        rmse_val = np.sqrt(mean_squared_error(y_test, test_pred))
        ax_i.scatter(y_test, test_pred, alpha=0.3, s=14, color=cmap(idx % 10))
        ax_i.plot([0, max_lim], [0, max_lim], "k--", linewidth=1.2)
        ax_i.set_title(f"{name}\nRMSE={rmse_val:.2f} | R$^2$={r2_val:.3f}", fontsize=9.5)
        ax_i.grid(True, alpha=0.3)
        if idx % n_cols == 0:
            ax_i.set_ylabel("Predicted [persons]")
        if idx >= (n_rows - 1) * n_cols:
            ax_i.set_xlabel("Observed [persons]")

    for extra_ax in axes_flat[n_models:]:
        extra_ax.set_visible(False)

    fig.suptitle("Predicted vs Observed Headcount Across All Regression Methods (Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_all_models_parity_grid.png"), dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot C: Bar chart comparison of MAE, RMSE, and R2 across all models
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(results_df))
    width = 0.36
    short_labels = [m.split(": ")[-1] if ": " in m else m for m in results_df["Model"]]

    axes[0].bar(x - width / 2, results_df["Test_MAE"], width, label="Test MAE [persons]", color="tab:blue")
    axes[0].bar(x + width / 2, results_df["Test_RMSE"], width, label="Test RMSE [persons]", color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, rotation=28, ha="right")
    axes[0].set_ylabel("Prediction Error [persons]")
    axes[0].set_title("Test MAE and RMSE by Regression Method")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, results_df["Val_R2"], width, label="Validation $R^2$", color="tab:green", alpha=0.75)
    axes[1].bar(x + width / 2, results_df["Test_R2"], width, label="Test $R^2$", color="tab:purple", alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=28, ha="right")
    axes[1].set_ylabel("Explained Variance ($R^2$)")
    axes[1].set_title("Validation and Test $R^2$ Comparison")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_model_metrics_comparison_bars.png"), dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot D: Cross-validation fold variation boxplot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [cv_fold_records[m] for m in results_df["Model"]]
    ax.boxplot(box_data, tick_labels=short_labels, showmeans=True)

    for pos, m_name in enumerate(results_df["Model"], start=1):
        fold_vals = cv_fold_records[m_name]
        ax.scatter(
            np.full(len(fold_vals), pos),
            fold_vals,
            color="tab:blue",
            edgecolor="white",
            s=45,
            zorder=3,
            label="Individual fold RMSE" if pos == 1 else None,
        )

    ax.set_ylabel("Cross-Validation RMSE [persons]")
    ax.set_title("5-Fold Time-Series Cross-Validation RMSE Variation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_cross_validation_fold_boxplots.png"), dpi=300)
    plt.close(fig)


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================

def run_pipeline(csv_path="Parsed_Data/synchronized_co2_occupancy_10min.csv", output_dir="Figures",
                 results_dir="results", causal_features=True):
    """
    Execute the complete training, validation, testing, and plotting workflow.

    Besides figures, every table printed to the console is written to ``results_dir``
    as CSV plus a ready-to-paste ``results_tables.md`` so README numbers are always
    traceable to a run.

    Parameters
    ----------
    csv_path : str
        Path to synchronized dataset CSV.
    output_dir : str
        Directory to save diagnostic and comparison figures.
    results_dir : str
        Directory to save CSV result tables, Markdown tables and run metadata.
    causal_features : bool, default=True
        Use causal (past-only) dC/dt features. Set False only to quantify how much
        future-sample leakage would inflate the scores.
    """
    if not os.path.exists(csv_path) and os.path.exists("synchronized_co2_occupancy_10min.csv"):
        csv_path = "synchronized_co2_occupancy_10min.csv"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    md_sections = []  # (title, DataFrame) pairs for results_tables.md
    train_df, val_df, test_df, df_326 = load_and_split_sensor_869(csv_path, causal_features=causal_features)

    target_col = "headcount_total_max" if "headcount_total_max" in train_df.columns else "occ_941f0352_max"

    print("=" * 78)
    print("STEP 0: NUMERICAL VERIFICATION OF DERIVATIVE STENCILS")
    print("=" * 78)
    p_centred = observed_convergence_order("centred")
    p_causal = observed_convergence_order("causal")
    print(f"Observed order, centred stencil (h = 2.0 -> 1.0 -> 0.5 min): {np.round(p_centred, 4)}")
    print(f"Observed order, causal stencil  (h = 2.0 -> 1.0 -> 0.5 min): {np.round(p_causal, 4)}")
    print(f"dC/dt features are {'CAUSAL (past-only)' if causal_features else 'NON-CAUSAL (reads future samples)'}.\n")

    print("=" * 78)
    print("STEP 1: CHRONOLOGICAL DATA SPLIT")
    print("=" * 78)
    print(f"Training Set   (70%): {len(train_df):5d} rows | CO2 range: [{train_df['co2_active'].min():.1f}, {train_df['co2_active'].max():.1f}] ppm")
    print(f"Validation Set (15%): {len(val_df):5d} rows | CO2 range: [{val_df['co2_active'].min():.1f}, {val_df['co2_active'].max():.1f}] ppm")
    print(f"Final Test Set (15%): {len(test_df):5d} rows | CO2 range: [{test_df['co2_active'].min():.1f}, {test_df['co2_active'].max():.1f}] ppm")
    print(f"Out-of-Regime (326) : {len(df_326):5d} rows | CO2 range: [{df_326['co2_active'].min():.1f}, {df_326['co2_active'].max():.1f}] ppm\n")

    # --------------------------------------------------------------------------
    # STEP 2: TIME-LAGGED CROSS-CORRELATION & FINITE-DIFFERENCE ANALYSIS
    # --------------------------------------------------------------------------
    lags_min, r_co2 = compute_lagged_correlations(
        train_df[target_col].values, train_df["co2_active"].values, max_lag_minutes=120, dt_minutes=10
    )
    _, r_dcdt = compute_lagged_correlations(
        train_df[target_col].values, train_df["dcdt_savgol"].values, max_lag_minutes=120, dt_minutes=10
    )
    best_lag_idx = int(np.argmax(r_co2))
    optimal_lag_min = lags_min[best_lag_idx]
    peak_r = r_co2[best_lag_idx]

    print("=" * 78)
    print("STEP 2: TIME-LAGGED CROSS-CORRELATION & DERIVATIVE ANALYSIS")
    print("=" * 78)
    print(f"Instantaneous Pearson r(tau=0 min)     : {r_co2[lags_min == 0][0]:.4f}")
    print(f"Optimal Physical Dispersion Lag tau*   : +{optimal_lag_min} minutes (Peak r = {peak_r:.4f})")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(lags_min, r_co2, "o-", label=r"CO$_2$ Concentration $C(t+\tau)$", linewidth=2)
    ax.plot(lags_min, r_dcdt, "s--", label=r"Smoothed Derivative $dC/dt$ (Savitzky-Golay)", linewidth=1.8)
    ax.axvline(optimal_lag_min, color="red", linestyle=":", label=f"Peak Lag $\\tau^* = {optimal_lag_min}$ min ($r={peak_r:.3f}$)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Time Lag $\tau$ (minutes) [Positive = CO$_2$ lags Occupancy]")
    ax.set_ylabel("Pearson Correlation $R(\\tau)$")
    ax.set_title("Time-Lagged Cross-Correlation on Training Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_lagged_cross_correlation.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # STEP 3: TRAIN, EVALUATE & PLOT ALL REGRESSION METHODS
    # --------------------------------------------------------------------------
    f_univariate = ["co2_active"]
    f_ode_mass = ["co2_active", "dcdt_savgol"]
    f_multivar = ["co2_active", "co2_lag1", "co2_lag2", "dcdt_savgol", "sin_time", "cos_time", "temp_c", "humidity_rh"]

    candidate_models = {
        "0. Baseline: Mean DummyRegressor": (
            f_univariate,
            DummyRegressor(strategy="mean")
        ),
        "0b. Engineering Baseline: CO2 > 600 ppm rule": (
            f_univariate,
            CO2ThresholdRule(threshold_ppm=600.0)
        ),
        "0c. Engineering Baseline: CO2 threshold tuned on train": (
            f_univariate,
            CO2ThresholdRule(tune=True)
        ),
        "1. Linear Regression: Static OLS": (
            f_univariate,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("ols", LinearRegression())])
        ),
        "2. Physical Mass-Balance ODE OLS": (
            f_ode_mass,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("ode_ols", LinearRegression())])
        ),
        "3. Regularized Ridge Regression": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        ),
        "4. Decision Tree Regressor": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("tree", DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42))])
        ),
        "5. Support Vector Regression (RBF)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("svr", SVR(kernel="rbf", C=10.0, epsilon=1.0, gamma="scale"))])
        ),
        "6. Neural Network (MLPRegressor)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=500, random_state=42))])
        ),
    }

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    tscv = TimeSeriesSplit(n_splits=5, gap=6)  # 6 bins = 1-hour purge buffer

    results_rows = []
    trained_pipelines = {}
    cv_fold_records = {}

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    for name, (feat_cols, pipeline) in candidate_models.items():
        X_tr = train_df[feat_cols].values
        X_va = val_df[feat_cols].values
        X_te = test_df[feat_cols].values

        t0 = timeit.default_timer()
        model = clone(pipeline)
        model.fit(X_tr, y_train)
        fit_ms = (timeit.default_timer() - t0) * 1000.0

        # Peak Python-heap allocation during fit (separate run: tracemalloc slows execution)
        tracemalloc.start()
        clone(pipeline).fit(X_tr, y_train)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_peak_mb = peak_bytes / 1e6

        val_pred = apply_physical_clipping(model.predict(X_va))
        # Inference time over the whole withheld test set: median of 10 repeats
        pred_ms = float(np.median(timeit.repeat(lambda: model.predict(X_te), repeat=10, number=1))) * 1000.0

        test_pred = apply_physical_clipping(model.predict(X_te))
        trained_pipelines[name] = (feat_cols, model, val_pred, test_pred)

        # 5-fold TimeSeriesSplit CV
        cv_rmses = []
        X_tv = train_val_df[feat_cols].values
        y_tv = train_val_df[target_col].values
        for tr_idx, va_idx in tscv.split(X_tv):
            cv_m = clone(pipeline)
            cv_m.fit(X_tv[tr_idx], y_tv[tr_idx])
            cv_p = apply_physical_clipping(cv_m.predict(X_tv[va_idx]))
            cv_rmses.append(np.sqrt(mean_squared_error(y_tv[va_idx], cv_p)))
        cv_fold_records[name] = cv_rmses

        val_metrics = evaluate_regression_metrics(y_val, val_pred)
        test_metrics = evaluate_regression_metrics(y_test, test_pred)

        # Save dedicated diagnostic figure for this individual regression method
        save_individual_model_plot(
            model_name=name,
            timestamps=test_df["timestamp_melbourne"],
            y_true=y_test,
            y_pred=test_pred,
            metrics_dict=test_metrics,
            output_dir=output_dir,
        )

        results_rows.append({
            "Model": name,
            "CV_RMSE": np.mean(cv_rmses),
            "Val_MAE": val_metrics["MAE_persons"],
            "Val_RMSE": val_metrics["RMSE_persons"],
            "Val_R2": val_metrics["R2"],
            "Test_MAE": test_metrics["MAE_persons"],
            "Test_RMSE": test_metrics["RMSE_persons"],
            "Test_R2": test_metrics["R2"],
            "Test_MAE_Empty": test_metrics["MAE_Empty"],
            "Test_MAE_Occ": test_metrics["MAE_Occupied"],
            "Fit_ms": fit_ms,
            "Pred_ms_testset": pred_ms,
            "Fit_peak_MB": fit_peak_mb,
        })

    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 78)
    print("STEP 3: MODEL COMPARISON TABLE")
    print("=" * 78)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    md_sections.append(("Model comparison (all models, chronological split)", results_df))

    # Save combined multi-model overlay, parity grid, metric bars, and CV boxplots
    save_combined_comparison_plots(
        test_df=test_df,
        target_col=target_col,
        trained_pipelines=trained_pipelines,
        results_df=results_df,
        cv_fold_records=cv_fold_records,
        output_dir=output_dir,
    )

    # --------------------------------------------------------------------------
    # STEP 3b: SELECTED MODEL vs ENGINEERING BASELINE (selection uses CV only)
    # --------------------------------------------------------------------------
    multivar_names = [n for n, (fc, _) in candidate_models.items() if fc == f_multivar]
    cv_lookup = results_df.set_index("Model")["CV_RMSE"]
    selected_name = cv_lookup.loc[list(candidate_models)].idxmin()
    best_multivar_name = cv_lookup.loc[multivar_names].idxmin()
    thr_name = "0c. Engineering Baseline: CO2 threshold tuned on train"
    rule_name = "0b. Engineering Baseline: CO2 > 600 ppm rule"
    thr_rmse = float(results_df.loc[results_df["Model"] == thr_name, "Test_RMSE"].iloc[0])

    cmp_names = ["0. Baseline: Mean DummyRegressor", rule_name, thr_name, selected_name]
    cmp_df = results_df[results_df["Model"].isin(cmp_names)][
        ["Model", "CV_RMSE", "Test_MAE", "Test_RMSE", "Test_R2", "Test_MAE_Empty", "Test_MAE_Occ",
         "Fit_ms", "Pred_ms_testset", "Fit_peak_MB"]].copy()
    cmp_df["Test_RMSE_change_vs_tuned_threshold_%"] = 100.0 * (cmp_df["Test_RMSE"] - thr_rmse) / thr_rmse

    print("\n" + "=" * 78)
    print("STEP 3b: SELECTED MODEL vs ENGINEERING (CO2-THRESHOLD) BASELINE")
    print("=" * 78)
    print(f"Selected by lowest 5-fold CV RMSE (no test data used): {selected_name}")
    print(cmp_df.to_string(index=False))
    cmp_df.to_csv(os.path.join(results_dir, "baseline_vs_selected.csv"), index=False)
    md_sections.append(("Selected model vs engineering baseline", cmp_df))

    # --------------------------------------------------------------------------
    # STEP 3c: FEATURE INFLUENCE (validation set) & TIME-FEATURE ABLATION
    # --------------------------------------------------------------------------
    all_groups = {
        "CO2 level & history": ["co2_active", "co2_lag1", "co2_lag2"],
        "CO2 rate dC/dt": ["dcdt_savgol"],
        "Time of day (sin/cos)": ["sin_time", "cos_time"],
        "Temperature & humidity": ["temp_c", "humidity_rh"],
    }
    groups = {k: [c for c in v if c in f_multivar] for k, v in all_groups.items()}
    bm_cols, bm_model, _, _ = trained_pipelines[best_multivar_name]
    imp_df, imp_base = grouped_permutation_importance(
        bm_model, val_df[bm_cols].values, y_val, bm_cols, groups, n_repeats=20, random_state=0
    )
    print("\n" + "=" * 78)
    print("STEP 3c: GROUPED PERMUTATION IMPORTANCE (validation set) & TIME ABLATION")
    print("=" * 78)
    print(f"Model: {best_multivar_name} | unpermuted validation RMSE = {imp_base:.3f} persons")
    print(imp_df.to_string(index=False))
    imp_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
    md_sections.append((f"Grouped permutation importance - {best_multivar_name} (validation set, RMSE increase in persons)", imp_df))

    time_cols = ["sin_time", "cos_time"]
    f_no_time = [c for c in f_multivar if c not in time_cols]
    f_co2_only = ["co2_active", "co2_lag1", "co2_lag2", "dcdt_savgol"]
    feature_sets = {"Full (8 features)": f_multivar,
                    "No time-of-day": f_no_time,
                    "CO2 physics only": f_co2_only}
    ablation_rows = []
    for name in multivar_names:
        for fs_label, fs_cols in feature_sets.items():
            pipe = candidate_models[name][1]
            m = clone(pipe).fit(train_df[fs_cols].values, y_train)
            va = evaluate_regression_metrics(y_val, m.predict(val_df[fs_cols].values))
            te = evaluate_regression_metrics(y_test, m.predict(test_df[fs_cols].values))
            cv = []
            for tr_idx, va_idx in tscv.split(train_val_df[fs_cols].values):
                cm = clone(pipe).fit(train_val_df[fs_cols].values[tr_idx], train_val_df[target_col].values[tr_idx])
                cp = apply_physical_clipping(cm.predict(train_val_df[fs_cols].values[va_idx]))
                cv.append(np.sqrt(mean_squared_error(train_val_df[target_col].values[va_idx], cp)))
            ablation_rows.append({"Model": name, "Feature_Set": fs_label, "N_features": len(fs_cols),
                                  "CV_RMSE": float(np.mean(cv)), "Val_RMSE": va["RMSE_persons"],
                                  "Val_R2": va["R2"], "Test_MAE": te["MAE_persons"],
                                  "Test_RMSE": te["RMSE_persons"], "Test_R2": te["R2"]})
    ablation_df = pd.DataFrame(ablation_rows)
    print("\nTime-of-day ablation (retrained from scratch for every feature set):")
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(results_dir, "time_feature_ablation.csv"), index=False)
    md_sections.append(("Feature-set ablation (does the model read the clock or sense CO2?)", ablation_df))

    fig_imp, (ax_i1, ax_i2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax_i1.barh(imp_df["Feature_Group"][::-1], imp_df["RMSE_increase_mean"][::-1],
               xerr=imp_df["RMSE_increase_std"][::-1], color="tab:blue", capsize=3)
    ax_i1.set_xlabel("Increase in validation RMSE when group is shuffled [persons]")
    ax_i1.set_title(f"Grouped Permutation Importance\n{best_multivar_name.split(': ')[-1]}", fontsize=10)
    ax_i1.grid(True, axis="x", alpha=0.3)
    pivot = ablation_df.pivot(index="Model", columns="Feature_Set", values="Test_RMSE")[list(feature_sets)]
    pivot.index = [m.split(". ", 1)[-1] for m in pivot.index]
    pivot.plot(kind="bar", ax=ax_i2, rot=20)
    ax_i2.set_ylabel("Test RMSE [persons]")
    ax_i2.set_title("Feature-Set Ablation (retrained per feature set)", fontsize=10)
    ax_i2.grid(True, axis="y", alpha=0.3)
    fig_imp.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_feature_importance_and_ablation.png"), dpi=300)
    plt.close(fig_imp)

    # --------------------------------------------------------------------------
    # STEP 4: OUT-OF-REGIME FAILURE MODE ANALYSIS ON SENSOR 326
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 4: OUT-OF-REGIME FAILURE MODE TEST ON CO2_SENSOR_326")
    print("=" * 78)
    if len(df_326) > 10 and target_col in df_326.columns:
        df_326_eval = df_326[df_326[target_col].notna()].copy()
        y_326 = df_326_eval[target_col].values
        r_326 = np.corrcoef(df_326_eval["co2_active"].values, y_326)[0, 1]
        print(f"Sensor 326 Physical Correlation r(CO2, Occupancy): {r_326:.4f}"
              + (" (sign inverted relative to the training regime)" if r_326 < 0 else ""))

        failure_rows = []
        fig_fail, axes_fail = plt.subplots(1, 2, figsize=(13, 5))

        # Left subplot: Physical regime shift scatter comparison (Sensor 869 vs Sensor 326)
        axes_fail[0].scatter(
            train_df["co2_active"], train_df[target_col],
            alpha=0.25, s=15, color="tab:blue", label="Sensor 869 (Training Regime)"
        )
        axes_fail[0].scatter(
            df_326_eval["co2_active"], y_326,
            alpha=0.25, s=15, color="tab:red", label=f"Sensor 326 (Out-of-Regime, r={r_326:.3f})"
        )
        axes_fail[0].set_xlabel(r"CO$_2$ Concentration [ppm]")
        axes_fail[0].set_ylabel("Observed Headcount [persons]")
        axes_fail[0].set_title("Physical Domain Shift: Sensor 869 vs Sensor 326")
        axes_fail[0].legend()
        axes_fail[0].grid(True, alpha=0.3)

        # Right subplot: Time-series failure on Sensor 326
        n_fail_win = min(len(df_326_eval), 500)
        t_326 = df_326_eval["timestamp_melbourne"].iloc[:n_fail_win]
        axes_fail[1].plot(t_326, y_326[:n_fail_win], color="black", linewidth=2, label="Observed Headcount (Sensor 326)")

        for name in ["0c. Engineering Baseline: CO2 threshold tuned on train",
                     "1. Linear Regression: Static OLS",
                     "2. Physical Mass-Balance ODE OLS",
                     "4. Decision Tree Regressor",
                     "5. Support Vector Regression (RBF)"]:
            f_cols, fitted_mod, _, _ = trained_pipelines[name]
            pred_326 = apply_physical_clipping(fitted_mod.predict(df_326_eval[f_cols].values))
            m_326 = evaluate_regression_metrics(y_326, pred_326)
            failure_rows.append({
                "Model": name,
                "Sensor326_MAE": m_326["MAE_persons"],
                "Sensor326_RMSE": m_326["RMSE_persons"],
                "Sensor326_R2": m_326["R2"],
            })
            axes_fail[1].plot(t_326, pred_326[:n_fail_win], linestyle="--", linewidth=1.4, label=name)

        axes_fail[1].set_xlabel("Melbourne Local Time")
        axes_fail[1].set_ylabel("Headcount [persons]")
        axes_fail[1].set_title("Out-of-Regime Prediction Failure on Sensor 326")
        axes_fail[1].legend(loc="upper right", fontsize=8.5)
        axes_fail[1].grid(True, alpha=0.3)

        fig_fail.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig6_out_of_regime_sensor326_failure.png"), dpi=300)
        plt.close(fig_fail)

        failure_df = pd.DataFrame(failure_rows)
        print(failure_df.to_string(index=False))
        failure_df.to_csv(os.path.join(results_dir, "out_of_regime_sensor326.csv"), index=False)
        md_sections.append(("Out-of-regime failure analysis (sensor 326)", failure_df))
        if (failure_df["Sensor326_R2"] < 0).any():
            print("-> Note: Negative R2 (< 0) means the model performs worse than predicting the mean,")
            print("   i.e. it does not transfer across an uncalibrated ventilation regime.")

    # --------------------------------------------------------------------------
    # STEP 5: TIMING & ARITHMETIC INTENSITY BENCHMARK
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5: NUMERICAL PROFILING, FLOPS & ARITHMETIC INTENSITY")
    print("=" * 78)
    perf_stats = benchmark_numerical_performance(train_df["co2_active"].values)
    for k, v in perf_stats.items():
        print(f"{k:38s}: {v:.4f}" if isinstance(v, float) else f"{k:38s}: {v}")
    perf_df = pd.DataFrame([perf_stats])
    perf_df.to_csv(os.path.join(results_dir, "performance_benchmark.csv"), index=False)
    md_sections.append(("Derivative benchmark (Python loop vs NumPy)", perf_df))

    # --------------------------------------------------------------------------
    # EXPORT: run metadata + ready-to-paste Markdown tables
    # --------------------------------------------------------------------------
    with open(os.path.join(results_dir, "run_metadata.txt"), "w") as fh:
        fh.write(f"python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
                 f"scikit-learn {sklearn.__version__}\n")
        fh.write(f"machine: {platform.platform()} | processor: {platform.processor() or 'unknown'}\n")
        fh.write(f"causal_features={causal_features} | selected model (lowest CV RMSE): {selected_name}\n")
        fh.write(f"observed order centred: {np.round(p_centred, 4).tolist()} | causal: {np.round(p_causal, 4).tolist()}\n")
    with open(os.path.join(results_dir, "results_tables.md"), "w") as fh:
        for title, frame in md_sections:
            fh.write(f"### {title}\n\n{df_to_markdown(frame)}\n\n")

    print(f"\nAll figures saved to '{output_dir}/'; all result tables saved to '{results_dir}/'.")


if __name__ == "__main__":
    run_pipeline()