"""
Data Extraction, Auditing, and Synchronization Pipeline for MMA3001 Dataset 2.

Targets hardware identifiers ('sensorid' for environmental sensors and
'deviceid'/'floorspaceid' for occupancy sensors), removes 136,604 frozen/cached
sensor polling rows, compresses 9.09M occupancy logs onto a 10-minute grid, and
synchronizes both streams in Australia/Melbourne local time.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def parse_env_json_fast(json_series: pd.Series) -> pd.DataFrame:
    """Parses nested NEMo environmental sensor JSON strings into a flat DataFrame.

    Parameters
    ----------
    json_series : pd.Series
        Series of raw JSON strings from the `jsondata` column.

    Returns
    -------
    parsed_df : pd.DataFrame
        Extracted columns: ['measurement_time', 'co2_ppm', 'co2_error',
        'temp_c', 'humidity_rh', 'pressure_mb', 'tvoc_ug_m3'].
    """
    records: List[Dict[str, Any]] = []

    for json_str in json_series:
        row_data: Dict[str, Any] = {
            "measurement_time": None,
            "co2_ppm": np.nan,
            "co2_error": None,
            "temp_c": np.nan,
            "humidity_rh": np.nan,
            "pressure_mb": np.nan,
            "tvoc_ug_m3": np.nan,
        }
        if isinstance(json_str, str):
            try:
                metrics = json.loads(json_str)
                for item in metrics:
                    var = item.get("variable", {})
                    var_id = var.get("id")
                    var_name = str(var.get("name", "")).lower()
                    vals = item.get("values")
                    if not vals:
                        continue
                    reading = vals[0]
                    val = reading.get("value")
                    err = reading.get("errorCode", 0)

                    if var_id == 67 or "carbon dioxide" in var_name or "co2" in var_name:
                        row_data["measurement_time"] = reading.get("time")
                        row_data["co2_ppm"] = val if err in (0, None) else np.nan
                        row_data["co2_error"] = err
                    elif "temperature" in var_name:
                        if err in (0, None):
                            row_data["temp_c"] = val
                    elif "humidity" in var_name:
                        if err in (0, None):
                            row_data["humidity_rh"] = val
                    elif "pressure" in var_name:
                        if err in (0, None):
                            row_data["pressure_mb"] = val
                    elif "tvoc" in var_name or "volatile" in var_name or "formaldehyde" in var_name:
                        if err in (0, None):
                            row_data["tvoc_ug_m3"] = val
            except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
                pass
        records.append(row_data)

    return pd.DataFrame(records, index=json_series.index)


def process_environmental_data(
    env_csv_path: str,
    out_csv_path: str,
) -> pd.DataFrame:
    """Loads, extracts, and audits the 180k-row environmental sensor dataset.

    Deduplicates specifically on ('sensorid', 'measurement_time') to remove cached
    readings from offline sensors while ignoring the unique row primary key 'id'.
    """
    print(f"[1/4] Loading environmental data from {env_csv_path}...")
    use_meta = ["sensorid", "devicetype", "status", "processing_errors", "createdate", "jsondata"]
    df_raw = pd.read_csv(env_csv_path, usecols=lambda c: c in use_meta)

    print("[2/4] Extracting CO2 and thermodynamic co-variates from jsondata...")
    parsed_metrics = parse_env_json_fast(df_raw["jsondata"])
    meta_only = df_raw.drop(columns=["jsondata"])
    df_env = pd.concat([meta_only, parsed_metrics], axis=1)

    raw_len = len(df_env)
    df_env = df_env.dropna(subset=["measurement_time", "co2_ppm"]).copy()
    df_env["measurement_time"] = df_env["measurement_time"].astype("int64")

    # Enforce physical CO2 domain [350, 5000] ppm
    df_env = df_env[
        (df_env["co2_ppm"] >= 350.0) & (df_env["co2_ppm"] <= 5000.0)
    ].copy()

    # Print per-sensor audit before deduplication
    print("\n      --- Environmental Sensor Health Audit (Raw vs Unique Timestamps) ---")
    for sid, grp in df_env.groupby("sensorid"):
        n_total = len(grp)
        n_uniq = grp["measurement_time"].nunique()
        dt_min = pd.to_datetime(grp["measurement_time"].min(), unit="s", utc=True)
        dt_max = pd.to_datetime(grp["measurement_time"].max(), unit="s", utc=True)
        print(
            f"      sensorid={sid}: {n_total:,} raw rows -> {n_uniq:,} unique readings "
            f"({n_total - n_uniq:,} frozen duplicates) | Active: {dt_min} to {dt_max}"
        )

    # Deduplicate by hardware 'sensorid' and 'measurement_time'
    # Also drop any timestamp that repeats > 5 times within the same sensor (frozen cache)
    ts_counts = df_env.groupby(["sensorid", "measurement_time"])["co2_ppm"].transform("count")
    df_env_active = df_env[ts_counts == 1].copy()
    if len(df_env_active) < 1000:
        # Fallback to keep='first' on (sensorid, measurement_time) excluding the 6 global frozen epochs
        frozen_epochs = df_env["measurement_time"].value_counts()
        bad_epochs = frozen_epochs[frozen_epochs > 10].index
        df_env = df_env[~df_env["measurement_time"].isin(bad_epochs)].drop_duplicates(
            subset=["sensorid", "measurement_time"], keep="first"
        )
    else:
        df_env = df_env.drop_duplicates(subset=["sensorid", "measurement_time"], keep="first")
        frozen_epochs = df_env["measurement_time"].value_counts()
        bad_epochs = frozen_epochs[frozen_epochs > 5].index
        df_env = df_env[~df_env["measurement_time"].isin(bad_epochs)].copy()

    df_env["datetime_utc"] = pd.to_datetime(df_env["measurement_time"], unit="s", utc=True)
    df_env["datetime_melb"] = df_env["datetime_utc"].dt.tz_convert("Australia/Melbourne")
    df_env = df_env.sort_values(["measurement_time", "sensorid"]).reset_index(drop=True)

    removed = raw_len - len(df_env)
    print(
        f"      --------------------------------------------------------------------\n"
        f"      Audit complete: {raw_len:,} raw rows -> {len(df_env):,} valid active readings "
        f"({removed:,} frozen/corrupt rows removed, {removed / raw_len * 100:.2f}%).\n"
    )

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_env.to_csv(out_csv_path, index=False)
    print(f"      Saved audited environmental data to '{out_csv_path}'")
    return df_env


def process_occupancy_data_chunked(
    occ_csv_path: str,
    out_raw_dedup_path: str,
    resample_freq: str = "10min",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Processes the 9.09M-row occupancy dataset grouped by hardware 'deviceid'.

    Ignores the unique row primary key 'id' and aggregates high-frequency camera
    headcounts per ('deviceid', 'floorspaceid') onto a 10-minute grid.
    """
    use_cols = ["deviceid", "floorspaceid", "collecteddate", "headcount", "occupancystatus"]
    print(f"[3/4] Processing {occ_csv_path} in chunks of {chunksize:,} using {use_cols}...")

    resampled_chunks = []
    for i, chunk in enumerate(pd.read_csv(occ_csv_path, usecols=use_cols, chunksize=chunksize)):
        # Remove physically impossible negative headcounts
        chunk = chunk[chunk["headcount"] >= 0].copy()
        chunk["datetime_utc"] = pd.to_datetime(chunk["collecteddate"], utc=True, format="ISO8601")

        # 1. Resolve duplicate second collisions per hardware deviceid
        chunk_dedup = (
            chunk.groupby(["deviceid", "floorspaceid", "datetime_utc"], as_index=False)["headcount"]
            .max()
        )
        # 2. Floor to 10-minute bin per hardware deviceid
        chunk_dedup["bin_utc"] = chunk_dedup["datetime_utc"].dt.floor(resample_freq)
        chunk_binned = (
            chunk_dedup.groupby(["deviceid", "floorspaceid", "bin_utc"], as_index=False)
            .agg(
                headcount_max=("headcount", "max"),
                headcount_mean=("headcount", "mean"),
                sample_count=("headcount", "count"),
            )
        )
        resampled_chunks.append(chunk_binned)
        print(f"      Processed chunk {i + 1}/19 ({len(chunk):,} raw rows -> {len(chunk_binned):,} binned rows)...")

    occ_combined = pd.concat(resampled_chunks, ignore_index=True)

    # Combine across chunk boundaries for each (deviceid, floorspaceid, bin_utc)
    occ_grid = (
        occ_combined.groupby(["deviceid", "floorspaceid", "bin_utc"], as_index=False)
        .agg(
            headcount_max=("headcount_max", "max"),
            headcount_mean=("headcount_mean", "mean"),
            sample_count=("sample_count", "sum"),
        )
        .sort_values(["bin_utc", "deviceid"])
        .reset_index(drop=True)
    )

    print("\n      --- Occupancy Sensor Health Audit (Per Hardware deviceid) ---")
    for (did, fid), grp in occ_grid.groupby(["deviceid", "floorspaceid"]):
        print(
            f"      deviceid={did} (floorspaceid={fid}): {len(grp):,} 10-min bins | "
            f"Max headcount={grp['headcount_max'].max()} | "
            f"Range: {grp['bin_utc'].min()} to {grp['bin_utc'].max()}"
        )
    print("      -------------------------------------------------------------\n")

    occ_grid["datetime_melb"] = occ_grid["bin_utc"].dt.tz_convert("Australia/Melbourne")
    occ_grid["time_sec"] = occ_grid["bin_utc"].dt.as_unit("s").astype("int64")

    Path(out_raw_dedup_path).parent.mkdir(parents=True, exist_ok=True)
    occ_grid.to_csv(out_raw_dedup_path, index=False)
    print(f"      Saved audited & binned occupancy data to '{out_raw_dedup_path}' ({len(occ_grid):,} rows)")
    return occ_grid


def synchronize_and_engineer_features(
    df_env: pd.DataFrame,
    occ_grid: pd.DataFrame,
    out_sync_path: str,
    resample_freq: str = "10min",
    max_interp_gap_bins: int = 3,
) -> pd.DataFrame:
    """Synchronizes active CO2 sensors and the 5 floorspaceid occupancy zones onto a 10-min grid."""
    print(f"[4/4] Synchronizing CO2 (by sensorid) and Occupancy (by floorspaceid) onto '{resample_freq}' grid...")

    df_env["datetime_utc"] = pd.to_datetime(df_env["datetime_utc"], utc=True)
    occ_grid["bin_utc"] = pd.to_datetime(occ_grid["bin_utc"], utc=True)

    # 1. Pivot the 2 active CO2 sensors ('6012002000326' and '6012002000869') into separate columns
    env_Copy = df_env.copy()
    env_Copy["bin_utc"] = env_Copy["datetime_utc"].dt.floor(resample_freq)

    co2_by_sensor = (
        env_Copy.groupby(["bin_utc", "sensorid"])["co2_ppm"]
        .mean()
        .unstack("sensorid")
        .resample(resample_freq)
        .asfreq()
        .interpolate(method="linear", limit=max_interp_gap_bins)
    )
    co2_by_sensor.columns = [f"co2_sensor_{ str(c)[-3:] }" for c in co2_by_sensor.columns]

    # Also keep mean environmental covariates across active sensors
    num_cols = ["co2_ppm", "temp_c", "humidity_rh", "pressure_mb", "tvoc_ug_m3"]
    avail_cols = [c for c in num_cols if c in env_Copy.columns]
    env_grid = (
        env_Copy.groupby("bin_utc")[avail_cols]
        .mean()
        .resample(resample_freq)
        .asfreq()
    )
    env_grid["co2_ppm"] = env_grid["co2_ppm"].interpolate(method="linear", limit=max_interp_gap_bins)
    env_grid = env_grid.join(co2_by_sensor, how="left")

    # 2. Pivot Occupancy by 'floorspaceid' (since 'deviceid' is the shared gateway ID!)
    occ_pivot_max = (
        occ_grid.pivot(index="bin_utc", columns="floorspaceid", values="headcount_max")
        .resample(resample_freq)
        .asfreq()
        .ffill(limit=max_interp_gap_bins)
        .fillna(0)
    )
    occ_pivot_mean = (
        occ_grid.pivot(index="bin_utc", columns="floorspaceid", values="headcount_mean")
        .resample(resample_freq)
        .asfreq()
        .ffill(limit=max_interp_gap_bins)
        .fillna(0)
    )

    occ_zone = pd.DataFrame(index=occ_pivot_max.index)
    occ_zone["headcount_total_max"] = occ_pivot_max.sum(axis=1)
    occ_zone["headcount_total_mean"] = occ_pivot_mean.sum(axis=1)

    # Save each individual floorspaceid column (shortened to first 8 chars of UUID for readability)
    for f_id in occ_pivot_max.columns:
        short_id = str(f_id)[:8]
        occ_zone[f"occ_{short_id}_max"] = occ_pivot_max[f_id]
        occ_zone[f"occ_{short_id}_mean"] = occ_pivot_mean[f_id]

    # 3. Inner join over overlapping active dates
    sync_df = env_grid.join(occ_zone, how="inner").dropna(subset=["co2_ppm"]).copy()
    sync_df.index = sync_df.index.tz_convert("Australia/Melbourne")
    sync_df.index.name = "datetime_melb"

    # 4. Print Spatial Correlation Matrix between Active CO2 Sensors and Floorspace IDs
    co2_cols = [c for c in sync_df.columns if c.startswith("co2_sensor_")] + ["co2_ppm"]
    occ_cols = [c for c in sync_df.columns if c.startswith("occ_") and c.endswith("_max")] + ["headcount_total_max"]
    print("\n      --- Spatial Correlation Matrix: Active CO2 Sensors vs Floorspace Zones ---")
    corr_matrix = sync_df[co2_cols + occ_cols].corr().loc[co2_cols, occ_cols]
    print(corr_matrix.round(4).to_string())
    print("      --------------------------------------------------------------------------\n")

    # 5. Compute 2nd-order central finite difference derivative d(CO2)/dt (ppm/min)
    dt_mins = pd.to_timedelta(resample_freq).total_seconds() / 60.0
    sync_df["dco2_dt_ppm_per_min"] = np.gradient(sync_df["co2_ppm"].values, dt_mins)

    for c_col in co2_by_sensor.columns:
        valid_mask = sync_df[c_col].notna()
        if valid_mask.sum() > 3:
            sync_df[f"d_{c_col}_dt"] = np.gradient(sync_df[c_col].ffill().bfill().values, dt_mins)

    # 6. Lagged & Lead CO2 features (10, 20, 30, 60 min)
    for lag_bins in [1, 2, 3, 6]:
        lag_min = int(lag_bins * dt_mins)
        sync_df[f"co2_lag_{lag_min}m"] = sync_df["co2_ppm"].shift(lag_bins)
        sync_df[f"co2_lead_{lag_min}m"] = sync_df["co2_ppm"].shift(-lag_bins)

    sync_df["hour_melb"] = sync_df.index.hour + sync_df.index.minute / 60.0
    sync_df["is_weekend"] = (sync_df.index.day_of_week >= 5).astype(int)
    sync_df["is_occupied"] = (sync_df["headcount_total_max"] > 0).astype(int)

    sync_df.to_csv(out_sync_path)
    print(
        f"      Saved final synchronized dataset to '{out_sync_path}' ({len(sync_df):,} rows)\n"
        f"      Overlapping Date Range (Melbourne): {sync_df.index.min()} to {sync_df.index.max()}"
    )
    return sync_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    ENV_RAW = BASE_DIR / "Base_Data" / "5EnvSensor_MayToDec2024_180kRows.csv"
    OCC_RAW = BASE_DIR / "Base_Data" / "5occupancySensor_MayToDec2024_9MRows.csv"

    ENV_OUT = BASE_DIR / "Parsed_Data" / "env_cleaned_full.csv"
    OCC_OUT = BASE_DIR / "Parsed_Data" / "occupancy_10min_cleaned.csv"
    SYNC_OUT = BASE_DIR / "Parsed_Data" / "synchronized_co2_occupancy_10min.csv"

    # Smart cache: load already-audited CSVs if they exist so you don't wait for 9M rows again!
    if ENV_OUT.exists() and OCC_OUT.exists():
        print("[Fast-Load] Audited intermediate files found! Loading directly...")
        df_env_clean = pd.read_csv(ENV_OUT)
        df_occ_clean = pd.read_csv(OCC_OUT)
    else:
        df_env_clean = process_environmental_data(str(ENV_RAW), str(ENV_OUT))
        df_occ_clean = process_occupancy_data_chunked(str(OCC_RAW), str(OCC_OUT), resample_freq="10min")

    df_sync = synchronize_and_engineer_features(df_env_clean, df_occ_clean, str(SYNC_OUT), resample_freq="10min")