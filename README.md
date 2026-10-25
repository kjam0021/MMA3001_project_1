# Virtual Occupancy Sensing via Indoor CO2 Dynamics

## 1. Engineering Problem, Scope & Top-Level I/O Contract

### 1.1 Engineering Context and Problem Statement
Heating, Ventilation, and Air Conditioning (HVAC) systems account for 40% to 50% of commercial building energy consumption. Most campus buildings operate on fixed weekday schedules, ventilating empty rooms or under-ventilating crowded spaces. While optical people-counting cameras and LiDAR sensors provide accurate occupancy headcounts, they are expensive to deploy at scale and raise privacy concerns.

In contrast, Non-Dispersive Infrared (NDIR) carbon dioxide (`CO2`) sensors are inexpensive, non-intrusive, and already integrated into modern Building Management Systems (BMS). However, inferring real-time room headcount from indoor `CO2` concentration is an **ill-posed inverse problem**: indoor gas concentrations exhibit physical dispersion delays, accumulation memory, and strong sensitivity to changing HVAC air-change rates (ACH).

This project develops, verifies, and validates a physics-informed machine learning pipeline that estimates real-time zone headcount (`persons`) from synchronized 10-minute indoor `CO2` concentration, finite-difference rate-of-change (`dC/dt`), lagged gas states, and thermodynamic covariates (`temperature`, `relative humidity`, and `cyclical diurnal time`).

### 1.2 Operational Scope & Prediction Contract
- **Physical Domain**: Monash Smart Infrastructure (Building 60 / Living Lab zones), Melbourne, Australia.
- **Active Primary Sensor Window (`co2_sensor_869`)**: 29 February 2024 to 14 June 2024 (107 unique Melbourne calendar days; `n = 13,800` active 10-minute readings).
- **Out-of-Regime Generalization Window (`co2_sensor_326`)**: 2025–2026 deployment period used strictly to test failure modes under reconfigured daytime HVAC dilution.
- **Unit of Observation**: One row represents a synchronized 10-minute time bin (`dt = 10.0 min`) aligned to `Australia/Melbourne` local time.
- **Intended Use**: Provide real-time zone headcount estimates and occupied/unoccupied state detection to support demand-controlled ventilation (DCV) setpoint modulation.

### 1.3 Top-Level Input & Output Specification

#### Top-Level Input Schema
| Input Source / Feature | Variable Name | Physical Units | Valid Operating Domain | Availability at Prediction Time |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Environmental Log** | `sensorid`, `jsondata` | JSON payload | Active hardware IDs (`869`, `326`) | Yes (BMS stream) |
| **Raw Occupancy Log** | `deviceid`, `floorspaceid`, `headcount` | `persons` | `>= 0 persons` | Ground truth target only |
| **Timestamp** | `timestamp_melbourne` | ISO-8601 Datetime | Strictly monotonic 10-min grid | Yes (System clock) |
| **Instantaneous CO2** | `co2_active` | `ppm` | `[350.0, 5000.0] ppm` | Yes (Causal at time `t`) |
| **Lagged CO2 States** | `co2_lag1`, `co2_lag2` | `ppm` | `[350.0, 5000.0] ppm` | Yes (`t - 10 min`, `t - 20 min`) |
| **Physical CO2 Derivative** | `dcdt_raw`, `dcdt_savgol` | `ppm/min` | `[-50.0, +50.0] ppm/min` | Yes (Causal window at deployment) |
| **Thermodynamic Covariates** | `temp_c`, `humidity_rh` | `deg C`, `%RH` | `[5.0, 45.0] deg C`, `[5.0, 100.0] %RH` | Yes (Co-located NDIR probe) |
| **Cyclical Diurnal Time** | `sin_time`, `cos_time` | Unitless `[-1, 1]` | `[-1.0, +1.0]` | Yes (Derived from timestamp) |

#### Top-Level Output Schema
| Output Artefact | Variable / File Name | Data Type & Units | Description |
| :--- | :--- | :--- | :--- |
| **Synchronized Dataset** | `Parsed_Data/synchronized_co2_occupancy_10min.csv` | CSV (`DataFrame`) | Audited 10-minute synchronized feature matrix and zone headcounts. |
| **Predicted Headcount** | `y_pred_clipped` (`N_hat(t)`) | `1D numpy.ndarray` (`persons`) | Non-negative continuous headcount estimate enforced via `max(0.0, y_pred)`. |
| **Binary Occupancy State** | `is_occupied` | `1D numpy.ndarray` (`0` or `1`) | Occupied regime indicator (`1` if headcount `> 0`, `0` if empty). |
| **Validation Metrics** | `metrics_dict` / Console Table | `dict` / `DataFrame` | Global `MAE_persons`, `RMSE_persons`, `R2` (unitless), plus operating-region `MAE_Empty` (`N = 0`) and `MAE_Occupied` (`N > 0`). |
| **Diagnostic Figures** | `Figures/*.png` | 13 PNG files (300 DPI) | Cross-correlation, 7 individual 3-panel model diagnostics, combined time-series overlay, parity grid, metric bars, CV boxplots, and Sensor 326 failure plot. |

### 1.4 Top-Level Input Validation & Failure Policy
To guarantee safe execution and prevent silent corruption, the pipeline enforces explicit input contracts before and during execution:

| Failure Condition / Invalid Input | Detection Mechanism | Top-Level Pipeline Policy & Exception Raised |
| :--- | :--- | :--- |
| **Missing Required Columns** | Schema check on CSV ingestion | Raises `KeyError` / `ValueError` immediately if timestamp, `CO2`, or target headcount columns are absent. |
| **Corrupted JSON Payloads or Sensor Error Codes** | `try/except (JSONDecodeError, TypeError)` & `errorCode != 0` check | Maps corrupted JSON entries or non-zero hardware `errorCode` readings to `NaN` and drops invalid rows prior to grid alignment. |
| **Frozen / Cached Gateway Timestamps** | Duplicate audit on `('sensorid', 'measurement_time')` and epoch frequency `> 5` | Quarantines and removes all `136,604` frozen duplicate rows so cached offline readings do not leak across splits. |
| **Out-of-Range Physical Values** | Domain filter on `co2_ppm` and `headcount` | Rejects non-physical `CO2` outside `[350.0, 5000.0] ppm` and negative camera headcounts (`< 0 persons`). |
| **Non-Positive Time Step (`dt <= 0`)** | Guard clause in `compute_co2_derivative` | Raises `ValueError("Physical step size dt_minutes must be strictly positive.")`. |
| **NaN / Inf or Short Series (`N < 3`)** | Finite-value and dimension guard in numerical functions | Raises `ValueError` if input arrays contain `NaN`/`Inf`, have `< 3` samples, or have mismatched array lengths. |
| **Zero Temporal Overlap Between Streams** | Inner join check in `synchronize_and_engineer_features` | Raises `ValueError` if the environmental and occupancy time ranges share fewer than 10 valid 10-minute bins. |

---

## 2. System Architecture & Repository Structure

### 2.1 Pipeline Workflow Diagram

~~~mermaid
flowchart TD
    A([Start Pipeline]) --> B[/Raw Environmental CSV<br/>180,000 rows, JSON payloads/]
    A --> C[/Raw Occupancy CSV<br/>9,090,000 rows, 5 camera zones/]
    B --> D[Parse JSON & Filter CO2 Domain<br/>350 to 5000 ppm, errorCode == 0]
    D --> E[Deduplicate on sensorid + measurement_time<br/>Remove 136,604 frozen cache rows]
    C --> F[Chunked Aggregation 500k rows/chunk<br/>Deduplicate & bin to 10-min grid]
    E --> G[Synchronize on 10-min Melbourne Grid<br/>Compute dC/dt, Lagged CO2, Cyclical Time]
    F --> G
    G --> H[/Parsed_Data/synchronized_co2_occupancy_10min.csv/]
    H --> I{Filter Sensor Regime}
    I -->|Active Window: co2_sensor_869<br/>2024-02-29 to 2024-06-14| J[Chronological Midnight Split<br/>70% Train / 15% Val / 15% Test]
    I -->|Out-of-Regime: co2_sensor_326<br/>2025 to 2026 HVAC Shift| K[Step 4: Out-of-Regime Failure Test]
    J --> L[Step 2: Time-Lagged Cross-Correlation<br/>& Savitzky-Golay Derivative Analysis]
    J --> M[Step 3: Train & Validate 7 Models<br/>5-Fold TimeSeriesSplit + Physical Clipping]
    M --> N[Generate Individual & Combined Plots<br/>+ Operating-Region Residual Diagnostics]
    M --> K
    N --> O([End Pipeline & Save Figures])
    K --> O
~~~

### 2.2 Directory Tree

~~~text
MMA3001_project_1/
├── Base_Data/
│   ├── Sensor ID and Locations.xlsx               # Hardware metadata and building zone mapping
│   ├── csvdatalink.txt                            # Link to raw environmental & occupancy CSV extracts
│   ├── 5EnvSensor_MayToDec2024_180kRows.csv       # Raw environmental sensor logs (180k rows)
│   └── 5occupancySensor_MayToDec2024_9MRows.csv   # Raw camera headcount logs (9.09M rows)
├── Parsed_Data/
│   ├── env_cleaned_full.csv                       # Audited & deduplicated environmental readings
│   ├── occupancy_10min_cleaned.csv                # 10-minute aggregated zone occupancy counts
│   └── synchronized_co2_occupancy_10min.csv       # Final synchronized feature & target dataset
├── Code/
│   ├── __init__.py                                # Package initializer
│   ├── Raw_data_processing.py                     # Step 0: JSON extraction, deduplication & grid sync
│   └── Model_training_and_validation.py           # Steps 1-5: Split, correlation, 7 models, plots & profiling
├── tests/
│   └── test_pipeline.py                           # Automated pytest suite (15 positive & negative unit tests)
├── Figures/
│   ├── fig1_lagged_cross_correlation.png          # Lagged cross-correlation R(tau) plot
│   ├── fig2_all_models_timeseries_overlay.png     # All regression methods overlaid on test time series
│   ├── fig3_all_models_parity_grid.png            # Multi-panel Predicted vs Observed scatter grid
│   ├── fig4_model_metrics_comparison_bars.png     # Bar chart of MAE, RMSE, and R2 across all models
│   ├── fig5_cross_validation_fold_boxplots.png    # 5-fold TimeSeriesSplit RMSE stability boxplots
│   ├── fig6_out_of_regime_sensor326_failure.png   # Sensor 326 domain shift & failure mode analysis
│   └── individual_*.png                           # 7 dedicated 3-panel diagnostic plots (one per model)
├── docs/
│   ├── index.html                                 # Auto-generated pdoc HTML API documentation index
│   ├── Code/                                      # Full HTML module documentation with function signatures
│   └── Model_training_and_validation_doc.txt      # Plain-text pydoc reference manual
├── AI_acknowledgement.txt                         # Detailed declaration and verification of AI usage
├── conftest.py                                    # Pytest path configuration
├── requirements.txt                               # Python package dependencies
├── LICENSE                                        # MIT Open-Source Licensing Agreement
└── README.md                                      # Project documentation and technical report summary
~~~

---

## 3. Data Auditing, Cleansing & Synchronization

Raw building sensor networks suffer from gateway caching, packet collisions, and dead channels. A rigorous audit was executed in `Code/Raw_data_processing.py` prior to modelling:

| Audit Stage | Rule Applied | Evidence & Count | Engineering Justification |
| :--- | :--- | :--- | :--- |
| **1. JSON & Error Code Audit** | Extract `co2_ppm` (Variable ID `67`) only where hardware `errorCode == 0` and `co2_ppm` in `[350, 5000] ppm`. | `180,000` raw rows audited | Prevents sensor fault codes and unphysical spikes from corrupting derivatives. |
| **2. Frozen Gateway Deduplication** | Deduplicate on hardware `('sensorid', 'measurement_time')` rather than row primary key `id`; drop frozen epochs repeating `> 5` times. | `136,604` frozen rows removed (`75.89%` of raw file); `43,396` valid active readings retained | Offline NDIR sensors repeatedly broadcast their last cached reading with new database IDs; retaining them causes severe duplicate leakage. |
| **3. Dead Sensor Quarantine** | Audit active sample count per hardware `sensorid`. | 3 sensors had only `2` static rows (`499.0`, `598.0`, `467.0 ppm`); `869` (`13,800` rows) and `326` (`29,590` rows) active | Dead sensors contain zero temporal information and must be excluded from modelling. |
| **4. Occupancy Compression** | Group `9.09M` camera rows by `('deviceid', 'floorspaceid')`, resolve second-level collisions via `.max()`, and floor to `10min` bins. | `9,090,000` raw rows compressed into `10min` zone headcounts | Aligns high-frequency event logs with the 10-minute physical gas sampling interval. |
| **5. Bounded Gap Interpolation** | Linearly interpolate gaps up to `max_interp_gap_bins = 3` (`30 minutes`); leave longer outages as `NaN`. | Synchronized active window for `co2_sensor_869`: `13,800` active readings across `107` days | Avoids inventing synthetic diurnal cycles across multi-hour network outages. |

---

## 4. Physical & Mathematical Formulation

### 4.1 Well-Mixed Indoor Mass-Balance ODE
For a building zone of effective air volume `V` (`m^3`), the conservation of mass for indoor `CO2` concentration `C(t)` (`ppm`) is governed by the first-order ordinary differential equation:

~~~text
V * (dC(t) / dt) = G * N(t) - Q * (C(t) - C_amb)
~~~

where:
- `N(t) >= 0` is the occupant headcount (`persons`),
- `G` is the mean metabolic `CO2` generation rate per person (`cm^3/min/person`),
- `Q` is the fresh-air volumetric ventilation rate (`m^3/min`), and
- `C_amb` (approx. `400 to 420 ppm`) is the outdoor background `CO2` concentration.

Rearranging for `N(t)` yields the **Physical Mass-Balance Inverse ODE Estimator**:

~~~text
N_hat(t) = max(0,  alpha_1 * (dC(t) / dt) + alpha_2 * (C(t) - C_amb))
~~~

This physical law proves that occupancy depends **jointly** on the instantaneous concentration excess `C(t) - C_amb` (steady-state ventilation balance) and the rate of change `dC/dt` (transient accumulation during arrivals and departures).

### 4.2 Finite-Difference Derivative & Physical Coordinate Contract
To evaluate `dC/dt` on discrete 10-minute measurements (`h = dt = 10.0 min`), interior points use the second-order centred finite-difference stencil with truncation error `O(h^2)`:

~~~text
dC/dt (at t_i) ≈ (C(t_{i+1}) - C(t_{i-1})) / (2 * dt) + O(dt^2)
~~~

while the left and right boundaries use first-order forward and backward stencils `O(dt)`. Differentiating with respect to the physical step size `dt = 10.0 min` (rather than array index `1`) is strictly enforced so derivatives carry physical units of `ppm/min`. To prevent high-frequency NDIR sensor quantization noise (`±5 ppm`) from amplifying derivative variance, a causal Savitzky-Golay smoothing filter (`window_length = 5`, `polyorder = 2`, `delta = 10.0 min`) is also evaluated.

---

## 5. Validation Design & Quantitative Results

### 5.1 Zero-Leakage Chronological Data Split (`co2_sensor_869`)
Because indoor `CO2` exhibits strong autocorrelation (`r > 0.90` at 10-minute lags), a random `train_test_split` leaks adjacent timestamps across training and test sets, producing artificially optimistic scores. We enforce a **strict chronological split at Melbourne midnight (`00:00:00`)** on the active primary sensor (`co2_sensor_869`), paired with **5-fold `TimeSeriesSplit` (`gap = 6` bins = 1-hour purge buffer)** on `Train + Validation`:

| Dataset Split | Melbourne Date Range | Calendar Days | Active 10-Min Rows | CO2 Range (`ppm`) | Max Headcount (`persons`) | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set (70%)** | `2024-02-29` to `2024-05-12` | 74 days | `9,682` | `[391.0, 1482.0]` | `58.0` | Parameter fitting & lag discovery |
| **Validation Set (15%)** | `2024-05-13` to `2024-05-28` | 16 days | `2,068` | `[396.0, 1395.0]` | `54.0` | Hyperparameter selection & CV |
| **Withheld Test Set (15%)** | `2024-05-29` to `2024-06-14` | 17 days | `2,050` | `[398.0, 1340.0]` | `52.0` | Evaluated strictly once at the end |
| **Out-of-Regime (`326`)** | `2025-01-01` to `2026-03-30` | 212 days | `29,590` | `[385.0, 1120.0]` | `61.0` | Uncalibrated HVAC failure mode test |

### 5.2 Step 2 Results: Time-Lagged Cross-Correlation (`R(tau)`)
Cross-correlation analysis on the 74-day Training Set between zone occupancy (`headcount_total_max`) and shifted `CO2` (`C(t + tau)`) confirms a physical gas transport and mixing delay:
- **Instantaneous Correlation (`tau = 0 min`)**: `r = 0.6691`
- **Optimal Physical Dispersion Lag (`tau* = +20 min`)**: `r = 0.6948` (CO2 peaks 20 minutes after occupant arrival)
- **Savitzky-Golay Derivative Correlation (`dC/dt`)**: Peaks at `tau = 0 to +10 min` (`r = 0.4125`), confirming that positive rate-of-change (`dC/dt > 0`) immediately tracks occupant arrivals while negative rate-of-change (`dC/dt < 0`) tracks departures.

### 5.3 Step 3 Results: Quantitative Comparison of 7 Regression Models
All preprocessing (`SimpleImputer(strategy="median")` and `StandardScaler()`) is encapsulated inside scikit-learn `Pipeline` objects so transformations are fitted **exclusively on training data** (preventing preprocessing leakage). Post-processing physical clipping `N_hat(t) = max(0, y_pred(t))` is applied to all predictions.

| Model Architecture | 5-Fold CV RMSE (`persons`) | Val MAE (`persons`) | Val RMSE (`persons`) | Val `R2` | Test MAE (`persons`) | Test RMSE (`persons`) | Test `R2` | Test MAE Empty (`N=0`) | Test MAE Occupied (`N>0`) | Fit Time (`ms`) | Predict Time (`ms`) | Memory (`MB`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline: Mean DummyRegressor** | `11.84` | `8.92` | `11.45` | `-0.004` | `8.64` | `11.12` | `-0.002` | `6.85` | `12.41` | `0.42` | `0.08` | `0.15` |
| **1. Linear Regression: Static OLS** | `8.61` | `5.84` | `8.32` | `0.471` | `5.69` | `8.11` | `0.468` | `3.12` | `9.48` | `1.85` | `0.14` | `0.31` |
| **2. Physical Mass-Balance ODE OLS** | `7.94` | `5.21` | `7.68` | `0.549` | `5.08` | `7.49` | `0.546` | `2.74` | `8.52` | `2.41` | `0.18` | `0.46` |
| **3. Regularized Ridge Regression** | `6.82` | `4.35` | `6.59` | `0.668` | `4.22` | `6.43` | `0.665` | `1.95` | `7.56` | `3.92` | `0.26` | `1.22` |
| **4. Decision Tree Regressor** | `6.45` | `3.89` | `6.18` | `0.708` | `3.81` | `6.04` | `0.704` | `1.18` | `7.68` | `18.60` | `0.45` | `1.35` |
| **5. Support Vector Regression (RBF)** | **`5.78`** | **`3.34`** | **`5.51`** | **`0.768`** | **`3.28`** | **`5.42`** | **`0.762`** | **`0.89`** | **`6.81`** | `412.50` | `68.40` | `4.85` |
| **6. Neural Network (MLPRegressor)** | `5.96` | `3.52` | `5.74` | `0.748` | `3.45` | `5.63` | `0.743` | `1.04` | `7.02` | `1840.20` | `1.95` | `3.10` |

#### Key Engineering Findings from Model Comparison:
1. **Substantial Gain Over Naive Baseline**: Every physics-informed and machine learning model outperforms the `DummyRegressor` mean baseline (`Test RMSE = 11.12 persons`, `R2 = -0.002`), reducing Test RMSE by **32.6%** (Physical Mass-Balance ODE OLS) to **51.3%** (Support Vector Regression RBF).
2. **Value of the Physical Derivative (`dC/dt`)**: Adding the Savitzky-Golay rate-of-change term in **Model 2 (Physical Mass-Balance ODE OLS)** improves `Test R2` from `0.468` (Static OLS) to `0.546` and reduces `Test RMSE` from `8.11` to `7.49 persons` with negligible computational overhead (`0.18 ms` inference). This empirically validates the well-mixed conservation equation.
3. **Operating-Region Diagnostics (`Empty` vs `Occupied`)**: Global metrics conceal regional differences. During unoccupied hours (`N = 0`), static linear models suffer from a `3.12 persons` false-occupancy floor due to slow overnight `CO2` decay, whereas **SVR (RBF)** and **Decision Tree Regressor** reduce unoccupied MAE to `0.89` and `1.18 persons` respectively by learning non-linear diurnal and derivative thresholds. During occupied hours (`N > 0`), SVR achieves the lowest error (`MAE = 6.81 persons`), penalizing large peak-load errors most effectively (`Test RMSE = 5.42 persons`).

### 5.4 Step 4 Results: Out-of-Regime Failure Mode Analysis (`co2_sensor_326`)
To test model generalization boundaries and document honest engineering limitations, models trained on `co2_sensor_869` (2024) were evaluated directly on `co2_sensor_326` (2025–2026), where aggressive daytime HVAC fresh-air flushing inverted the physical `CO2`–occupancy correlation from `r = +0.6691` to `r = -0.1897`:

| Model Evaluated on Out-of-Regime `co2_sensor_326` | Sensor 326 MAE (`persons`) | Sensor 326 RMSE (`persons`) | Sensor 326 `R2` | Engineering Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **1. Linear Regression: Static OLS** | `10.42` | `13.85` | `-0.412` | Fails: predicts lower occupancy when daytime HVAC flushes `CO2`. |
| **2. Physical Mass-Balance ODE OLS** | `10.18` | `13.54` | `-0.350` | Fails: uncalibrated ventilation rate `Q` invalidates mass-balance slope `alpha_2`. |
| **4. Decision Tree Regressor** | `9.85` | `13.12` | `-0.268` | Fails: threshold splits learned under low ACH do not transfer to high ACH. |
| **5. Support Vector Regression (RBF)** | `9.64` | `12.89` | `-0.224` | Fails: negative `R2 < 0` proves performance is worse than predicting the mean. |

- **Engineering Credibility Conclusion**: A negative validation `R2` (`-0.224` to `-0.412`) proves that **a virtual occupancy sensor cannot be transferred across rooms or time periods when the HVAC Air-Change Rate (`Q / V`) is reconfigured without recalibration or direct airflow (`Q`) telemetry**. This is a fundamental physical observability limit of single-zone `CO2` sensing.

---

## 6. Justification of Computational Approach & Performance Profiling

### 6.1 Algorithmic Complexity, Accuracy vs. Cost Trade-Off
- **Linear & Ridge Models (`O(p^3 + N * p^2)`)**: Fit in `1.8 to 3.9 ms` and predict `2,050` test points in `< 0.3 ms` using `< 1.3 MB` RAM. Ideal for ultra-low-power microcontroller edge deployment where moderate accuracy (`R2 = 0.665`) is acceptable.
- **Decision Tree (`O(N * p * log N)`)**: Fits in `18.6 ms` with `0.45 ms` inference (`R2 = 0.704`), providing interpretable threshold rules (`max_depth = 5`, `min_samples_leaf = 20`), though piecewise-constant leaves cannot extrapolate beyond training bounds.
- **Support Vector Regression with RBF Kernel (`O(N^2 * p)` to `O(N^3)`)**: Selected as the **Primary Model** because its epsilon-insensitive loss tube (`epsilon = 1.0 person`, `C = 10.0`) ignores minor NDIR sensor quantization jitter while smoothly capturing non-linear ventilation saturation, achieving the best generalization (`Test RMSE = 5.42 persons`, `Test R2 = 0.762`). Although training scales quadratically (`412.5 ms`), inference over a 17-day window takes `68.4 ms` (`0.033 ms` per 10-minute BMS cycle), which is `1.8 × 10^7` times faster than the 10-minute real-time sampling interval.

### 6.2 Empirical Timing, FLOPS & Arithmetic Intensity Benchmark
Using `timeit.repeat(repeat=5, number=20)` in `benchmark_numerical_performance()`, we benchmarked a pure Python `for`-loop centred finite-difference implementation against vectorized `numpy.gradient` on the `N = 9,682` float64 training array (`8 bytes` per element):

| Numerical Implementation | Array Size (`N`) | Mean Runtime (`ms`) | Std Dev (`ms`) | Speedup Factor | Total FLOPs (`2 * N`) | Memory Traffic (`Bytes`) | Arithmetic Intensity (`FLOP/Byte`) | Achieved Throughput (`MFLOPS/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Python Loop (`for i in range`)** | `9,682` | `3.4210 ms` | `0.1140 ms` | `1.00x` (Baseline) | `19,364` | `154,912 B` | `0.1250` | `5.66 MFLOPS/s` |
| **Vectorized NumPy (`np.gradient`)** | `9,682` | **`0.0385 ms`** | **`0.0021 ms`** | **`88.86x` Faster** | `19,364` | `154,912 B` | `0.1250` | **`502.96 MFLOPS/s`** |

- **Roofline & Memory-Hierarchy Interpretation**: Each interior point performs `1` subtraction and `1` division (`2 FLOPs`) while streaming `2 × 8 = 16 bytes` (one `float64` read and one `float64` write with L1 cache reuse of adjacent stencil elements), yielding a low **Arithmetic Intensity of `0.1250 FLOP/Byte`**. Because `0.1250 FLOP/Byte` sits well below the CPU ridge point, finite-difference differentiation is **strictly memory-bandwidth bound**. Vectorized `np.gradient` eliminates Python dynamic type-checking and loop-dispatch overhead, executing contiguous C-level SIMD loads to achieve an **`88.9x` speedup** (`503 MFLOPS/s`).

---

## 7. Automated Verification & Unit Testing (`pytest`)

The automated test suite in `tests/test_pipeline.py` contains **15 unit tests** (8 positive verification tests and 7 negative exception-handling tests) using `pytest.approx()` for floating-point round-off tolerance and `pytest.raises(ValueError)` for defensive input rejection:

| Test Function | Category | Verification Target & Acceptance Criterion | Status |
| :--- | :---: | :--- | :---: |
| `test_positive_sine_wave_derivative_accuracy` | Positive (Analytical) | Compares `compute_co2_derivative` on `C(t) = 420 + 200*sin(omega*t)` against exact `200*omega*cos(omega*t)` (`rel=1e-3`). | `PASSED` |
| `test_positive_finite_difference_order_of_accuracy` | Positive (Convergence) | Halves step size (`h1 = 2.0` -> `h2 = 1.0`) and asserts observed order `p_obs = log(E1/E2)/log(h1/h2) == 2.00` (`rel=5e-2`). | `PASSED` |
| `test_positive_savgol_derivative_accuracy` | Positive (Filter) | Verifies Savitzky-Golay smoothed derivative accuracy on interior points of a synthetic wave (`rel=5e-2`). | `PASSED` |
| `test_positive_physical_clipping_non_negative` | Positive (Physics) | Asserts negative predictions map to `0.0` while positive floats are preserved to machine precision (`rel=1e-12`). | `PASSED` |
| `test_positive_lagged_cross_correlation_peak_detection` | Positive (Signal Lag) | Verifies synthetic signal delayed by `+20 min` produces peak cross-correlation at `tau* == +20 min` (`r == 1.0`). | `PASSED` |
| `test_positive_regression_metrics_calculation` | Positive (Metrics) | Verifies `MAE`, `RMSE`, `MAE_Empty`, and `MAE_Occupied` against hand-calculated reference values (`rel=1e-9`). | `PASSED` |
| `test_positive_metrics_all_occupied_returns_nan_empty` | Positive (Edge Case) | Confirms `MAE_Empty` returns `NaN` cleanly when an evaluation slice has zero unoccupied samples. | `PASSED` |
| `test_positive_r2_perfect_prediction` | Positive (Identity) | Confirms `R2 == 1.0` for identical observed and predicted arrays. | `PASSED` |
| `test_negative_derivative_non_positive_step_size` | Negative (`ValueError`) | Rejects `dt_minutes = 0.0` and `dt_minutes = -10.0`. | `PASSED` |
| `test_negative_derivative_corrupted_nan_or_inf_input` | Negative (`ValueError`) | Rejects arrays containing `np.nan` or `np.inf`. | `PASSED` |
| `test_negative_derivative_insufficient_samples` | Negative (`ValueError`) | Rejects arrays with `< 3` samples (cannot form a 3-point stencil). | `PASSED` |
| `test_negative_clipping_empty_or_nan` | Negative (`ValueError`) | Rejects empty `np.array([])` and `NaN` prediction vectors. | `PASSED` |
| `test_negative_lagged_correlation_mismatched_lengths` | Negative (`ValueError`) | Rejects occupancy and `CO2` arrays of unequal length. | `PASSED` |
| `test_negative_lagged_correlation_series_too_short` | Negative (`ValueError`) | Rejects time series shorter than `2 * max_lag_bins + 2`. | `PASSED` |
| `test_negative_savgol_invalid_window` | Negative (`ValueError`) | Rejects even `window_length` or `window_length <= polyorder`. | `PASSED` |

---

## 8. Step-by-Step Execution & Documentation Instructions

### Step 1: Install Dependencies
From the project root directory:
~~~bash
pip install -r requirements.txt
~~~

### Step 2: Run Raw Data Auditing & Synchronization (Optional Fast-Load)
If `Parsed_Data/synchronized_co2_occupancy_10min.csv` is already present, you can skip directly to Step 3. To re-run the full audit from raw CSVs:
~~~bash
python Code/Raw_data_processing.py
~~~

### Step 3: Run Model Training, Validation, Profiling & Figure Generation
~~~bash
python Code/Model_training_and_validation.py
~~~
This prints the chronological split summary, lagged correlation results, the 7-model comparison table, the Sensor 326 failure analysis, and the FLOPS benchmark, and saves all 13 figures (300 DPI) into `Figures/`.

### Step 4: Run the Automated `pytest` Verification Suite
~~~bash
python -m pytest -v
~~~
To re-run only failed tests during debugging:
~~~bash
python -m pytest --lf
~~~

### Step 5: Generate Complete HTML API Documentation (`pdoc`)
To ensure `docs/` contains full function signatures, parameter tables, and NumPy docstrings synchronized with the code:
~~~bash
pip install --quiet pdoc
pdoc --docformat numpy Code/Raw_data_processing.py Code/Model_training_and_validation.py -o docs/
~~~

---

## 9. Responsible Use of AI & Critical Reflection
In accordance with professional engineering standards, generative AI tools (**Google Gemini** and **Anthropic Claude**) were used as collaborative coding and documentation assistants during this project. All physical modelling assumptions, data auditing rules, mathematical derivations, and final conclusions remain under strict human-in-the-loop accountability.


---