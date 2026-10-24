import json
import pandas as pd

def extract_co2(json_str):
    """Extracts the CO2 reading, timestamp, and any error code from jsondata."""
    try:
        metrics = json.loads(json_str)
        for item in metrics:
            var = item.get("variable", {})
            # Match by variable id (67) or name ("Carbon dioxide")
            if var.get("id") == 67 or var.get("name") == "Carbon dioxide":
                 reading = item.get("values", [{}])[0]
                 return pd.Series([
                     reading.get("time"),
                     reading.get("value"),
                      var.get("unit"),
                     reading.get("errorCode")
                 ])
    except (json.JSONDecodeError, TypeError, IndexError):
        pass
    return pd.Series([None, None, None, None])

# 1. Load the CSV file
df = pd.read_csv("Project_1/MMA3001_project_1/Base_Data/5EnvSensor_MayToDec2024_180kRows.csv")

# 2. Extract CO2 details into new columns
df[["measurement_time", "co2_value", "unit", "error_code"]] = df["jsondata"].apply(extract_co2)

# 3. Convert Unix epoch time to a readable UTC datetime
df["measurement_datetime"] = pd.to_datetime(df["measurement_time"], unit="s", utc=True)

# 4. Select only the relevant columns
time_ppm_df = df[["measurement_time", "co2_value"]]
# Save just those two columns to a new CSV
time_ppm_df.to_csv("Project_1/MMA3001_project_1/Parsed_Data/co2_time_ppm.csv", index=False)

occ_df = pd.read_csv(
    "Project_1/MMA3001_project_1/Base_Data/5occupancySensor_MayToDec2024_9MRows.csv",
    usecols=["collecteddate", "headcount", "occupancystatus"]
)

# Convert the timezone-aware string (e.g., '2023-11-24 00:36:52+11') to UTC datetime & Unix seconds
occ_df["datetime_utc"] = pd.to_datetime(occ_df["collecteddate"], utc=True)
occ_df["time"] = occ_df["datetime_utc"].astype("int64") // 10**9

# Keep just the time and headcount (add "occupancystatus" to the list if needed)
occ_clean = occ_df[["time", "datetime_utc", "headcount"]]

# Save Occupancy data
occ_clean.to_csv("Project_1/MMA3001_project_1/Parsed_Data/occupancy_extracted.csv", index=False)
print("Saved Occupancy data to 'occupancy_extracted.csv'")