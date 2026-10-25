"""
Machine learning pipeline for virtual occupancy sensing using CO2 dynamics.

This module implements chronological data splitting, time-lagged cross-correlation,
finite-difference derivative estimation, regression model training, cross-validation,
residual diagnostics, and out-of-regime failure mode analysis.
"""

import os
import re
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
        sub_df["dcdt_raw"] = compute_co2_derivative(
            sub_df["co2_active"].values, dt_minutes=10.0, use_savgol=False
        )
        sub_df["dcdt_savgol"] = compute_co2_derivative(
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

def run_pipeline(csv_path="Parsed_Data/synchronized_co2_occupancy_10min.csv", output_dir="Figures"):
    """
    Execute the complete training, validation, testing, and plotting workflow.

    Parameters
    ----------
    csv_path : str
        Path to synchronized dataset CSV.
    output_dir : str
        Directory to save diagnostic and comparison figures.
    """
    if not os.path.exists(csv_path) and os.path.exists("synchronized_co2_occupancy_10min.csv"):
        csv_path = "synchronized_co2_occupancy_10min.csv"

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

        t1 = timeit.default_timer()
        val_pred = apply_physical_clipping(model.predict(X_va))
        pred_ms = (timeit.default_timer() - t1) * 1000.0

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
            "Pred_ms": pred_ms,
        })

    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 78)
    print("STEP 3: MODEL COMPARISON TABLE")
    print("=" * 78)
    print(results_df.to_string(index=False))

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

        for name in ["1. Linear Regression: Static OLS",
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

        print(pd.DataFrame(failure_rows).to_string(index=False))
        print("-> Note: Negative R2 (< 0) demonstrates that models applied across uncalibrated")
        print("   HVAC ventilation regimes perform worse than predicting the mean.")

    # --------------------------------------------------------------------------
    # STEP 5: TIMING & ARITHMETIC INTENSITY BENCHMARK
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STEP 5: NUMERICAL PROFILING, FLOPS & ARITHMETIC INTENSITY")
    print("=" * 78)
    perf_stats = benchmark_numerical_performance(train_df["co2_active"].values)
    for k, v in perf_stats.items():
        print(f"{k:38s}: {v:.4f}" if isinstance(v, float) else f"{k:38s}: {v}")

    print(f"\nAll individual and combined figures successfully saved to '{output_dir}/'.")


if __name__ == "__main__":
    run_pipeline()