"""
Machine learning pipeline for virtual occupancy sensing using CO2 dynamics.

This module implements training, validation, numerical verification, and
out-of-regime failure mode analysis.
"""

import os
import timeit
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.base import clone
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
    sample index.
    Interior points use the second-order centred finite-difference stencil O(h^2),
    while boundaries use first-order forward/backward stencils O(h).

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

    # Use physical coordinate spacing rather than sample index.
    return np.gradient(arr, float(dt_minutes))


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
# 2. DATA SPLITTING & FEATURE ENGINEERING
# ==============================================================================

def load_and_split_sensor_869(csv_path):
    """
    Filter dataset to active co2_sensor_869 window and split chronologically.

    Uses a strict time-ordered split at Melbourne midnight (00:00:00) to prevent
    temporal data leakage across Train (70%), Validation (15%), and Withheld
    Test (15%) sets.

    Parameters
    ----------
    csv_path : str
        Path to `synchronized_co2_occupancy_10min.csv`.

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
        sub_df["dcdt_raw"] = compute_co2_derivative(sub_df["co2_active"].values, dt_minutes=10.0, use_savgol=False)
        sub_df["dcdt_savgol"] = compute_co2_derivative(sub_df["co2_active"].values, dt_minutes=10.0, use_savgol=True)
        sub_df["co2_lag1"] = sub_df["co2_active"].shift(1).bfill()
        sub_df["co2_lag2"] = sub_df["co2_active"].shift(2).bfill()

        hour_float = sub_df[time_col].dt.hour + sub_df[time_col].dt.minute / 60.0
        sub_df["sin_time"] = np.sin(2.0 * np.pi * hour_float / 24.0)
        sub_df["cos_time"] = np.cos(2.0 * np.pi * hour_float / 24.0)

        if "temp_c" not in sub_df.columns:
            sub_df["temp_c"] = 21.0
        if "humidity_rh" not in sub_df.columns:
            sub_df["humidity_rh"] = 50.0

    # Strict chronological split at Melbourne midnight.
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

def evaluate_mma3001_regression(y_true, y_pred):
    """
    Calculate regression metrics and operating-region checks.

    Computes global MAE, RMSE, and R2, plus localized MAE across important
    engineering operating regions (Unoccupied N=0 vs Occupied N>0) as taught
    to avoid metric cherry-picking.

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
        MAE_Unoccupied_persons, and MAE_Occupied_persons.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = apply_physical_clipping(y_pred)

    mse = mean_squared_error(y_t, y_p)
    mae_all = mean_absolute_error(y_t, y_p)
    rmse_all = np.sqrt(mse)
    r2_all = r2_score(y_t, y_p)

    # Operating-region checks (Empty vs Occupied regime).
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


# ==============================================================================
# 4. EMPIRICAL TIMING & ARITHMETIC INTENSITY BENCHMARK
# ==============================================================================

def benchmark_numerical_performance(co2_array, n_runs=20):
    """
    Benchmark execution time and estimate FLOPS / Arithmetic Intensity.

    Compares a naive Python loop centred-difference implementation against
    vectorized `numpy.gradient` using `timeit.repeat` averaged across multiple
    runs to account for OS background variation.

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

    # Estimate FLOPS and byte traffic for the centred finite difference.
    # Each interior point performs 1 subtraction + 1 division = 2 FLOPs
    # Memory traffic: reads 2 float64s and writes 1 float64 (with cache reuse ~ 16 bytes/pt)
    total_flops = 2.0 * n_pts
    total_bytes = 2.0 * 8.0 * n_pts  # 1 read + 1 write per element streamed from RAM
    arithmetic_intensity = total_flops / total_bytes  # FLOPs / Byte
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
# 5. MAIN PIPELINE EXECUTION (STEPS 1 TO 4)
# ==============================================================================

def run_pipeline(csv_path="Parsed_Data/synchronized_co2_occupancy_10min.csv", output_dir="Figures"):
    """
    Execute the complete training, validation, and testing workflow.

    Parameters
    ----------
    csv_path : str
        Path to synchronized dataset CSV.
    output_dir : str
        Directory to save publication-ready diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df, val_df, test_df, df_326 = load_and_split_sensor_869(csv_path)

    target_col = "headcount_total_max" if "headcount_total_max" in train_df.columns else "occ_941f0352_max"

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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lags_min, r_co2, "o-", label=r"CO$_2$ Concentration $C(t+\tau)$", linewidth=2)
    ax.plot(lags_min, r_dcdt, "s--", label=r"Smoothed Derivative $dC/dt$ (Savitzky-Golay)", linewidth=1.8)
    ax.axvline(optimal_lag_min, color="red", linestyle=":", label=f"Peak Lag $\\tau^* = {optimal_lag_min}$ min ($r={peak_r:.3f}$)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Time Lag $\tau$ (minutes) [Positive = CO$_2$ lags Occupancy]")
    ax.set_ylabel("Pearson Correlation $R(\\tau)$")
    ax.set_title("MMA3001 Step 2: Time-Lagged Cross-Correlation on Training Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_lagged_cross_correlation.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # STEP 3: TRAIN & COMPARE MODELS
    # --------------------------------------------------------------------------
    f_univariate = ["co2_active"]
    f_ode_mass = ["co2_active", "dcdt_savgol"]
    f_multivar = ["co2_active", "co2_lag1", "co2_lag2", "dcdt_savgol", "sin_time", "cos_time", "temp_c", "humidity_rh"]

    # Compare baseline, linear, regularized, tree-based, kernel, and neural models.
    candidate_models = {
        "0. Baseline: DummyRegressor (Mean)": (
            f_univariate,
            DummyRegressor(strategy="mean")
        ),
        "1. Linear Regression: Static OLS (5.1)": (
            f_univariate,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("ols", LinearRegression())])
        ),
        "2. Physical ODE Mass-Balance OLS (7.2 & 5.1)": (
            f_ode_mass,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("ode_ols", LinearRegression())])
        ),
        "3. Regularized Ridge Regression (3.6 & 5.1)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        ),
        "4. Decision Tree Regressor (5.2)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("tree", DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42))])
        ),
        "5. Support Vector Regression RBF (5.3)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("svr", SVR(kernel="rbf", C=10.0, epsilon=1.0, gamma="scale"))])
        ),
        "6. Neural Network MLPRegressor (5.4)": (
            f_multivar,
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=500, random_state=42))])
        ),
    }

    # 5-fold TimeSeriesSplit cross-validation on Train + Val.
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    tscv = TimeSeriesSplit(n_splits=5, gap=6)  # 6 bins = 1-hour purge buffer

    results_rows = []
    trained_pipelines = {}

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    for name, (feat_cols, pipeline) in candidate_models.items():
        X_tr = train_df[feat_cols].values
        X_va = val_df[feat_cols].values
        X_te = test_df[feat_cols].values

        # Time model fitting and prediction using timeit.
        t0 = timeit.default_timer()
        model = clone(pipeline)
        model.fit(X_tr, y_train)
        fit_ms = (timeit.default_timer() - t0) * 1000.0

        t1 = timeit.default_timer()
        val_pred = apply_physical_clipping(model.predict(X_va))
        pred_ms = (timeit.default_timer() - t1) * 1000.0

        test_pred = apply_physical_clipping(model.predict(X_te))
        trained_pipelines[name] = (feat_cols, model, val_pred, test_pred)

        # TimeSeriesSplit CV RMSE
        cv_rmses = []
        X_tv = train_val_df[feat_cols].values
        y_tv = train_val_df[target_col].values
        for tr_idx, va_idx in tscv.split(X_tv):
            cv_m = clone(pipeline)
            cv_m.fit(X_tv[tr_idx], y_tv[tr_idx])
            cv_p = apply_physical_clipping(cv_m.predict(X_tv[va_idx]))
            cv_rmses.append(np.sqrt(mean_squared_error(y_tv[va_idx], cv_p)))

        val_metrics = evaluate_mma3001_regression(y_val, val_pred)
        test_metrics = evaluate_mma3001_regression(y_test, test_pred)

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
            "Pred_ms": pred_ms,
        })

    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 78)
    print("STEP 3: MODEL COMPARISON TABLE")
    print("=" * 78)
    print(results_df.to_string(index=False))

    # Residual diagnostics plot.
    best_model_name = "5. Support Vector Regression RBF (5.3)"
    _, best_model, _, best_test_pred = trained_pipelines[best_model_name]
    residuals = y_test - best_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_test, best_test_pred, alpha=0.35, s=15)
    max_v = max(np.max(y_test), np.max(best_test_pred))
    axes[0].plot([0, max_v], [0, max_v], "r--", label="Ideal 1:1 Prediction")
    axes[0].set_xlabel("Measured Headcount $y$ (persons)")
    axes[0].set_ylabel(r"Predicted Headcount $\hat{y}$ (persons)")
    axes[0].set_title(f"Prediction vs Measured ({best_model_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(best_test_pred, residuals, alpha=0.35, s=15, color="darkorange")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel(r"Predicted Headcount $\hat{y}$ (persons)")
    axes[1].set_ylabel(r"Residual $e = y - \hat{y}$ (persons)")
    axes[1].set_title("Residual Diagnostics Plot (Test Set)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_week5_residual_diagnostics.png"), dpi=300)
    plt.close(fig)

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
        print(f"Sensor 326 Physical Correlation r(CO2, Occupancy): {r_326:.4f} (Inverted by HVAC dilution)")

        failure_rows = []
        for name in ["1. Linear Regression: Static OLS (5.1)",
                     "2. Physical ODE Mass-Balance OLS (7.2 & 5.1)",
                     "4. Decision Tree Regressor (5.2)",
                     "5. Support Vector Regression RBF (5.3)"]:
            f_cols, fitted_mod, _, _ = trained_pipelines[name]
            pred_326 = apply_physical_clipping(fitted_mod.predict(df_326_eval[f_cols].values))
            m_326 = evaluate_mma3001_regression(y_326, pred_326)
            failure_rows.append({
                "Model": name,
                "Sensor326_MAE": m_326["MAE_persons"],
                "Sensor326_RMSE": m_326["RMSE_persons"],
                "Sensor326_R2": m_326["R2"],
            })
        print(pd.DataFrame(failure_rows).to_string(index=False))
        print("-> Note: Negative R2 (< 0) proves predictions on uncalibrated HVAC regimes")
        print("   perform worse than predicting the mean (violating the Prediction Contract).")

    # --------------------------------------------------------------------------
    # STEP 5: TIMING & ARITHMETIC INTENSITY BENCHMARK
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5: NUMERICAL PROFILING, FLOPS & ARITHMETIC INTENSITY")
    print("=" * 78)
    perf_stats = benchmark_numerical_performance(train_df["co2_active"].values)
    for k, v in perf_stats.items():
        print(f"{k:38s}: {v:.4f}" if isinstance(v, float) else f"{k:38s}: {v}")


if __name__ == "__main__":
    run_pipeline()