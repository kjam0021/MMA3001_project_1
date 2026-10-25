# Virtual Occupancy Sensing via Indoor CO2 Dynamics and Machine Learning

An end-to-end engineering computation, numerical differentiation, and machine learning regression pipeline for estimating building zone headcount from indoor carbon dioxide (CO2) concentration dynamics and thermodynamic covariates.

## 1. Project Overview

Direct camera-based occupancy counting systems are expensive to deploy at scale and raise occupant privacy concerns. Conversely, non-dispersive infrared (NDIR) environmental CO2 sensors are inexpensive and widely integrated into modern Heating, Ventilation, and Air Conditioning (HVAC) Building Management Systems (BMS).

However, inferring real-time occupancy headcount from indoor CO2 concentration is an inverse problem governed by physical gas dispersion delay, ventilation dilution, and sensor polling artefacts. This project implements a reproducible two-stage computational pipeline that:

1. **Audits and Synchronizes Multi-Modal Sensor Streams (`Code/Raw_data_processing.py`):**

   * Parses nested JSON payloads from over 180,000 environmental sensor records (`sensorid`) and processes 9.09 million high-frequency occupancy camera logs (`deviceid` and `floorspaceid`).

   * Identifies and removes 136,604 frozen or cached offline sensor polling duplicates and restricts CO2 readings to the physically valid domain of 350 to 5000 ppm.

   * Aligns active hardware sensors (`co2_sensor_869` and `co2_sensor_326`) with 5 monitored floor-space zones onto a uniform 10-minute temporal grid in `Australia/Melbourne` local time (`Parsed_Data/synchronized_co2_occupancy_10min.csv`).

2. **Models, Validates, and Audits Virtual Occupancy Estimators (`Code/Model_training_and_validation.py`):**

   * Enforces a zero-leakage chronological split at Melbourne midnight (`00:00:00`) across the active `co2_sensor_869` campaign (107 calendar days, 13,800 active readings): **70% Training** (74 days), **15% Validation** (16 days), and **15% Withheld Test** (17 days), alongside a 5-fold `TimeSeriesSplit` cross-validator with a 1-hour purge buffer (`gap=6`).

   * Quantifies the physical gas transport delay via time-lagged Pearson cross-correlation across lags from -120 to +120 minutes and estimates physical rates of change (`dC/dt`) using second-order centred finite differences and Savitzky-Golay smoothing filters.

   * Benchmarks 7 regression architectures across global metrics (MAE, RMSE, R-squared), operating-region checks (Unoccupied vs. Occupied regimes), and empirical runtime and FLOPS performance.

   * Evaluates out-of-regime failure modes on `co2_sensor_326` (2025-2026), demonstrating quantitatively how uncalibrated changes in HVAC air-change rates (ACH) invert the physical CO2-occupancy correlation.

## 2. Pipeline Architecture and Workflow

The following Mermaid flowchart illustrates the complete data processing, feature engineering, model training, and verification workflow:

```
flowchart TD
    subgraph Stage1["Stage 1: Raw Data Auditing & Synchronization (Code/Raw_data_processing.py)"]
        A[/"Raw Environmental Logs\n(180k rows, nested JSON)"/] --> B["Parse JSON & Filter Physical Domain\n350 to 5000 ppm"]
        C[/"Raw Occupancy Camera Logs\n(9.09M rows, 5 floorspace zones)"/] --> D["Chunked Processing (500k rows/chunk)\nRemove Negative Counts & Floor to 10-min Bins"]
        B --> E["Hardware Sensor Health Audit\nRemove 136,604 Frozen/Cached Polling Rows"]
        E --> F["Pivot Active Sensors (869 & 326)\n& Synchronize on 10-min Melbourne Grid"]
        D --> F
        F --> G[/"Synchronized Dataset\nParsed_Data/synchronized_co2_occupancy_10min.csv"/]
    end

    subgraph Stage2["Stage 2: Numerical Differentiation, Training & Validation (Code/Model_training_and_validation.py)"]
        G --> H["Filter Primary Active Window (co2_sensor_869)\n& Out-of-Regime Window (co2_sensor_326)"]
        H --> I["Compute Physical Derivatives (dC/dt)\nCentred Finite Difference & Savitzky-Golay Filter"]
        I --> J["Chronological Split at 00:00:00 Melbourne Time\nTrain (70%) | Val (15%) | Test (15%)"]
        J --> K["Time-Lagged Cross-Correlation\nIdentify Optimal Dispersion Delay"]
        J --> L["Train & Cross-Validate 7 Regression Models\n5-Fold TimeSeriesSplit (gap = 6 bins)"]
        L --> M["Apply Physical Non-Negativity Clipping\nClipped Prediction = max(0, Raw Prediction)"]
        M --> N["Evaluate Global & Operating-Region Metrics\nMAE, RMSE, R2, Empty vs. Occupied MAE"]
        M --> O["Out-of-Regime Failure Mode Test\nEvaluate Trained Models on co2_sensor_326"]
        N --> P[/"Save Diagnostic Figures\nFigures/*.png"/]
        O --> P
    end

    subgraph Stage3["Stage 3: Automated Verification (tests/test_pipeline.py)"]
        Q["Pytest Positive & Negative Unit Tests\nAnalytical Sine-Wave Convergence & Exception Checks"] -.-> I
        Q -.-> M
        Q -.-> N
    end

```

## 3. Repository Structure

```
MMA3001_project_1/
├── .gitignore                                    # Ignores large raw CSVs, __pycache__, and virtual environments
├── AI_acknowledgement.txt                        # Declaration of AI usage and human-in-the-loop verification
├── MMA3001_2026_Project_Brief.pdf                # Official project specification and assessment rubric
├── README.md                                     # Project documentation, setup, and execution guide
├── conftest.py                                   # Pytest root path configuration
├── requirements.txt                              # Python package dependencies
├── test_report.html                              # Standalone HTML report of automated pytest execution
│
├── Base_Data/                                    # Raw sensor metadata and external dataset links
│   ├── Sensor ID and Locations.xlsx              # Hardware sensor spatial mapping and building floorplans
│   ├── csvdatalink.txt                           # Download link for the raw environmental & occupancy CSVs
│   ├── 5EnvSensor_MayToDec2024_180kRows.csv      # (Downloaded locally via csvdatalink.txt)
│   └── 5occupancySensor_MayToDec2024_9MRows.csv  # (Downloaded locally via csvdatalink.txt)
│
├── Parsed_Data/                                  # Cleaned and synchronized datasets generated by Stage 1
│   ├── env_cleaned_full.csv                      # Deduplicated active environmental sensor readings
│   ├── occupancy_10min_cleaned.csv               # 10-minute aggregated zone headcounts
│   └── synchronized_co2_occupancy_10min.csv      # Final synchronized feature matrix in Melbourne time
│
├── Code/                                         # Core Python source package
│   ├── __init__.py                               # Package initialization
│   ├── Raw_data_processing.py                    # Stage 1: JSON parsing, sensor auditing, and 10-min synchronization
│   └── Model_training_and_validation.py          # Stage 2: Differentiation, lag analysis, ML training, and plotting
│
├── Figures/                                      # Publication-ready diagnostic figures generated by Stage 2
│   ├── fig1_lagged_cross_correlation.png         # Time-lagged Pearson cross-correlation plot
│   ├── fig2_all_models_timeseries_overlay.png    # Multi-model time-series tracking comparison
│   ├── fig3_all_models_parity_grid.png           # Side-by-side Predicted vs. Observed parity grid
│   ├── fig4_model_metrics_comparison_bars.png    # Grouped bar charts of MAE, RMSE, and R2 across models
│   ├── fig5_cross_validation_fold_boxplots.png   # 5-fold TimeSeriesSplit RMSE distribution boxplots
│   ├── fig6_out_of_regime_sensor326_failure.png  # Out-of-regime HVAC dilution failure analysis on Sensor 326
│   └── individual_*.png                          # Dedicated 3-panel diagnostic reports for each regression model
│
├── docs/                                         # Auto-generated API documentation (pdoc & pydoc)
│   ├── index.html                                # HTML documentation landing page
│   ├── search.js                                 # Client-side API documentation search index
│   ├── Model_training_and_validation_doc.txt     # Plain-text pydoc reference manual
│   └── Code/
│       ├── Model_training_and_validation.html    # HTML API docs for Model_training_and_validation.py
│       └── Raw_data_processing.html              # HTML API docs for Raw_data_processing.py
│
└── tests/                                        # Automated unit test suite
    └── test_pipeline.py                          # Positive and negative pytest unit tests

```

## 4. Physical and Numerical Principles

* **Indoor Mass-Balance Dynamics:** Indoor CO2 accumulation depends on the balance between occupant metabolic generation and fresh-air HVAC ventilation dilution. Consequently, static CO2 concentration alone lags behind occupancy changes, whereas combining instantaneous CO2 (`co2_active`) with its rate of change (`dC/dt`) captures both transient occupant arrival (`dC/dt > 0`) and departure (`dC/dt < 0`).

* **Finite-Difference Derivative Estimation:** Discrete sensor measurements are sampled at uniform physical intervals (`dt = 10.0 minutes`). Interior points use a second-order centred finite-difference stencil (`numpy.gradient(co2, dt)`), while boundary points use first-order forward and backward differences.

* **Savitzky-Golay Filtering:** To suppress high-frequency NDIR sensor quantization noise without distorting diurnal trends, a local polynomial Savitzky-Golay filter (`scipy.signal.savgol_filter`, window length 5, polynomial order 2, `delta = 10.0 minutes`) is applied prior to differentiation.

* **Physical Non-Negativity Constraint:** Because building headcount cannot be negative, all raw regression outputs are post-processed via `apply_physical_clipping` (`np.maximum(0.0, predictions)`).

## 5. Installation and Dataset Setup

### 5.1 Clone the Repository and Install Dependencies

Ensure you have **Python 3.9+** installed. From your local terminal (or in a Google Colab cell prefixed with `!`), run:

```
# Clone the repository and navigate to the project root
git clone https://github.com/kjam0021/mma3001_project_1.git
cd mma3001_project_1

# Install required packages (numpy, pandas, scipy, scikit-learn, matplotlib, pytest, pdoc)
pip install -r requirements.txt

```

### 5.2 Raw Data Placement

Because the raw occupancy log (`5occupancySensor_MayToDec2024_9MRows.csv`) exceeds GitHub's file size limit, download the raw CSV files using the URL provided in `Base_Data/csvdatalink.txt` and place them inside the `Base_Data/` directory:

* `Base_Data/5EnvSensor_MayToDec2024_180kRows.csv`

* `Base_Data/5occupancySensor_MayToDec2024_9MRows.csv`

*(Note: If `Parsed_Data/synchronized_co2_occupancy_10min.csv` is already present in your workspace, `Code/Model_training_and_validation.py` can be executed immediately without re-running the 9.09-million-row chunked parser.)*

## 6. Execution Instructions

Always execute commands from the **project root directory**.

### Step 1: Run Raw Data Auditing and 10-Minute Grid Synchronization

To parse the raw JSON environmental logs, deduplicate frozen sensor timestamps, aggregate the 9.09 million occupancy records in 500,000-row chunks, and build the synchronized dataset:

```
python Code/Raw_data_processing.py

```

* **Outputs:**

  * Prints the hardware sensor health audit (raw vs. unique active readings) and spatial correlation matrix across floor-space zones.

  * Saves `Parsed_Data/env_cleaned_full.csv`, `Parsed_Data/occupancy_10min_cleaned.csv`, and `Parsed_Data/synchronized_co2_occupancy_10min.csv`.

### Step 2: Run Model Training, Validation, Failure Analysis, and Plotting

To execute the chronological data split, time-lagged cross-correlation analysis, 7-model training and validation benchmark, out-of-regime failure test on `co2_sensor_326`, and FLOPS profiling:

```
python Code/Model_training_and_validation.py

```

* **Outputs:**

  * Prints Step 1-5 summary tables directly to the console (chronological split bounds, optimal lag, full 7-model comparison table across CV/Val/Test sets, Sensor 326 failure table, and `timeit` / FLOPS benchmarks).

  * Saves all individual model diagnostic reports (`individual_*.png`) and combined multi-model comparison figures (`fig1_*.png` through `fig6_*.png`) to `Figures/`.

### Step 3: Run the Automated `pytest` Verification Suite

To execute all 15 positive and negative unit tests verifying numerical differentiation accuracy, second-order convergence, physical non-negativity clipping, cross-correlation peak detection, and exception handling:

```
python -m pytest -v

```

* To re-run only the last failed tests during debugging:

  ```
  python -m pytest --lf
  
  ```

* An archived standalone test execution report is also available at `test_report.html`.

### Step 4: View or Regenerate API Documentation (`pdoc` and `pydoc`)

Pre-built HTML and plain-text documentation files are stored in the `docs/` directory (`docs/index.html`). To regenerate them after modifying docstrings:

```
# Generate interactive HTML API documentation
pdoc ./Code -o ./docs

# Generate plain-text reference documentation using built-in pydoc
python -m pydoc Code.Model_training_and_validation > docs/Model_training_and_validation_doc.txt

```

## 7. Candidate Models and Validation Strategy

### 7.1 Zero-Leakage Chronological Splitting

Random k-fold splitting on autocorrelated time-series data leaks future concentration trajectories into the training set. Therefore, `co2_sensor_869` (`2024-02-29` to `2024-06-14`) is partitioned strictly at `00:00:00` Melbourne time:

* **Training Set (First 70% - 74 calendar days):** `2024-02-29` to `2024-05-12`

* **Validation Set (Next 15% - 16 calendar days):** `2024-05-13` to `2024-05-28`

* **Withheld Test Set (Final 15% - 17 calendar days):** `2024-05-29` to `2024-06-14` (evaluated strictly once)

All preprocessing steps (`SimpleImputer(strategy="median")` and `StandardScaler()`) are encapsulated inside `sklearn.pipeline.Pipeline` objects so that summary statistics are fitted exclusively on training partitions.

### 7.2 Evaluated Regression Approaches

| **Model ID and Name** | **Feature Set** | **Computational Structure** | 
| **0. Baseline: `DummyRegressor`** | `co2_active` | Predicts the constant training mean headcount to establish a zero-skill reference (`R2 = 0`). | 
| **1. Static Univariate OLS** | `co2_active` | Ordinary Least Squares linear model using instantaneous CO2 concentration only. | 
| **2. Physical Mass-Balance ODE OLS** | `co2_active`, `dcdt_savgol` | Inverse lumped mass-balance linear estimator incorporating the smoothed rate of change (`dC/dt`). | 
| **3. Regularized Ridge Regression** | Multivariate (8 features)\* | L2-penalized linear regression (`alpha = 10.0`) mitigating multicollinearity among lagged CO2 features. | 
| **4. Decision Tree Regressor** | Multivariate (8 features)\* | Axis-aligned recursive partitioning (`max_depth=5`, `min_samples_leaf=20`) capturing non-linear thresholds. | 
| **5. Support Vector Regression (RBF)** | Multivariate (8 features)\* | Radial Basis Function kernel (`C=10.0`, `epsilon=1.0`) modelling smooth non-linear gas saturation and diurnal cycles. | 
| **6. Neural Network (`MLPRegressor`)** | Multivariate (8 features)\* | Two-hidden-layer multilayer perceptron `(32, 16)` with L2 weight decay (`alpha=0.01`). | 

*\*Multivariate feature set (8 features):* `co2_active`, `co2_lag1` (10-min lag), `co2_lag2` (20-min lag), `dcdt_savgol`, `sin_time`, `cos_time`, `temp_c`, and `humidity_rh`.

## 8. Summary of Automated Unit Tests (`tests/test_pipeline.py`)

The test suite (`tests/test_pipeline.py`) implements **8 positive tests** and **7 negative tests**:

* **Positive Verification Tests:**

  * `test_positive_sine_wave_derivative_accuracy`: Verifies that `compute_co2_derivative` matches the exact analytical derivative on a synthetic harmonic sine wave within `pytest.approx(rel=1e-3)`.

  * `test_positive_finite_difference_order_of_accuracy`: Halves the grid step size from `h1 = 2.0` to `h2 = 1.0` minutes and asserts that the observed order of accuracy (`p_obs = log(E(h1) / E(h2)) / log(h1 / h2)`) converges to `2.0` within 5% tolerance (confirming second-order truncation error reduction by a factor of 4).

  * `test_positive_savgol_derivative_accuracy`: Verifies interior accuracy of the Savitzky-Golay derivative filter on a smooth synthetic profile.

  * `test_positive_physical_clipping_non_negative`: Verifies that `apply_physical_clipping` maps negative predictions to `0.0` while preserving positive floating-point values to `1e-12` relative tolerance.

  * `test_positive_lagged_cross_correlation_peak_detection`: Verifies exact recovery of a synthetic +20-minute phase delay (`peak_lag = 20 min`, `r = 1.0`).

  * `test_positive_regression_metrics_calculation`, `test_positive_metrics_all_occupied_returns_nan_empty`, and `test_positive_r2_perfect_prediction`: Verify exact calculation of global MAE, RMSE, R-squared, and operating-region metrics (`MAE_Empty`, `MAE_Occupied`).

* **Negative Exception-Handling Tests (`pytest.raises(ValueError)`):**

  * Rejects non-positive time step sizes (`dt_minutes <= 0`), preventing division-by-zero or inverted time coordinates.

  * Rejects corrupted `NaN` or `Inf` sensor arrays in both derivative and clipping routines.

  * Rejects arrays with fewer than 3 samples (insufficient for a 3-point centred stencil).

  * Rejects mismatched array lengths or overly short time series in lagged cross-correlation.

  * Rejects invalid Savitzky-Golay filter configurations (even `window_length` or `window_length <= polyorder`).

## 9. AI Acknowledgment

In accordance with engineering software governance and academic integrity guidelines, generative AI tools were used as collaborative assistants during the development of this repository for code structuring, docstring formatting, and unit test scaffolding. All mathematical derivations, data auditing rules, chronological split boundaries, and numerical outputs were independently verified and validated by the author. Full details are documented in [`AI_acknowledgement.txt`](AI_acknowledgement.txt).