# Virtual Occupancy Sensing via Indoor CO2 Dynamics

## 1. Engineering Problem, Scope & Top-Level I/O Contract

### 1.1 Engineering Context and Problem Statement
Heating, Ventilation, and Air Conditioning (HVAC) systems account for 40% to 50% of commercial building energy consumption. Most campus buildings operate on fixed weekday schedules, ventilating empty rooms or under-ventilating crowded spaces. While optical people-counting cameras and LiDAR sensors provide accurate occupancy headcounts, they are expensive to deploy at scale and raise privacy concerns.

In contrast, Non-Dispersive Infrared (NDIR) carbon dioxide (`CO2`) sensors are inexpensive, non-intrusive, and already integrated into modern Building Management Systems (BMS). However, inferring real-time room headcount from indoor `CO2` concentration is an **ill-posed inverse problem**: indoor gas concentrations exhibit physical dispersion delays, accumulation memory, and strong sensitivity to changing HVAC air-change rates (ACH).

This project develops, verifies, and validates a physics-informed machine learning pipeline that estimates real-time zone headcount (`persons`) from synchronized 10-minute indoor `CO2` concentration, finite-difference rate-of-change (`dC/dt`), lagged gas states, and thermodynamic covariates (`temperature`, `relative humidity`, and `cyclical diurnal time`).

### 1.2 Operational Scope & Prediction Contract
- **Physical Domain**: Monash Smart Infrastructure (Building 60 / Living Lab zones), Melbourne, Australia.
- **Active Primary Sensor Window (`co2_sensor_869`)**: 29 February 2024 to 14 June 2024 (107 unique Melbourne calendar days; `n = 15,194` synchronized 10-minute bins used for modelling, i.e. 98.6% of the 15,408 bins possible in 107 days).
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
| **Physical CO2 Derivative** | `dcdt_raw`, `dcdt_savgol` | `ppm/min` | `[-50.0, +50.0] ppm/min` | Yes (trailing stencils using only `C(t)`, `C(t-10)`, ... via `compute_causal_co2_derivative`) |
| **Thermodynamic Covariates** | `temp_c`, `humidity_rh` | `deg C`, `%RH` | `[5.0, 45.0] deg C`, `[5.0, 100.0] %RH` | Yes (Co-located NDIR probe) |
| **Cyclical Diurnal Time** | `sin_time`, `cos_time` | Unitless `[-1, 1]` | `[-1.0, +1.0]` | Yes (Derived from timestamp) |

#### Top-Level Output Schema
| Output Artefact | Variable / File Name | Data Type & Units | Description |
| :--- | :--- | :--- | :--- |
| **Synchronized Dataset** | `Parsed_Data/synchronized_co2_occupancy_10min.csv` | CSV (`DataFrame`) | Audited 10-minute synchronized feature matrix and zone headcounts. |
| **Predicted Headcount** | `y_pred_clipped` (`N_hat(t)`) | `1D numpy.ndarray` (`persons`) | Non-negative continuous headcount estimate enforced via `max(0.0, y_pred)`. |
| **Binary Occupancy State** | `is_occupied` | `1D numpy.ndarray` (`0` or `1`) | Occupied regime indicator (`1` if headcount `> 0`, `0` if empty). |
| **Validation Metrics** | `metrics_dict` / Console Table | `dict` / `DataFrame` | Global `MAE_persons`, `RMSE_persons`, `R2` (unitless), plus operating-region `MAE_Empty` (`N = 0`) and `MAE_Occupied` (`N > 0`). |
| **Diagnostic Figures** | `Figures/*.png` | 16 PNG files (300 DPI) | Cross-correlation, 9 individual 3-panel model diagnostics (7 models + 2 CO2-threshold baselines), combined time-series overlay, parity grid, metric bars, CV boxplots, Sensor 326 failure plot, and feature-importance/ablation plot. |
| **Result Tables** | `results/*.csv`, `results/results_tables.md`, `results/run_metadata.txt` | CSV / Markdown | Every table quoted in this README, written by the run that produced it (model comparison, baseline vs selected, feature importance, ablation, out-of-regime, benchmark) plus library versions and the causal-feature flag. |

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
    J --> M[Step 3: Train & Validate 9 Models<br/>incl. 2 CO2-threshold baselines<br/>5-Fold TimeSeriesSplit + Physical Clipping]
    M --> N[Generate Individual & Combined Plots<br/>+ Feature Importance & Time Ablation]
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
│   └── Model_training_and_validation.py           # Steps 0-5: Verification, split, correlation, 9 models, importance/ablation, plots & profiling
├── tests/
│   └── test_pipeline.py                           # Automated pytest suite (24 tests: 14 positive, 10 negative)
├── Figures/
│   ├── fig1_lagged_cross_correlation.png          # Lagged cross-correlation R(tau) plot
│   ├── fig2_all_models_timeseries_overlay.png     # All regression methods overlaid on test time series
│   ├── fig3_all_models_parity_grid.png            # Multi-panel Predicted vs Observed scatter grid
│   ├── fig4_model_metrics_comparison_bars.png     # Bar chart of MAE, RMSE, and R2 across all models
│   ├── fig5_cross_validation_fold_boxplots.png    # 5-fold TimeSeriesSplit RMSE stability boxplots
│   ├── fig6_out_of_regime_sensor326_failure.png   # Sensor 326 domain shift & failure mode analysis
│   ├── fig7_feature_importance_and_ablation.png   # Grouped permutation importance + time-feature ablation
│   └── individual_*.png                           # 9 dedicated 3-panel diagnostic plots (one per model)
├── results/
│   ├── model_comparison.csv                       # All 9 models: CV/val/test metrics, fit/predict time, peak memory
│   ├── baseline_vs_selected.csv                   # Dummy vs CO2-threshold rules vs selected model
│   ├── feature_importance.csv                     # Grouped permutation importance (validation set)
│   ├── time_feature_ablation.csv                  # Full vs no-time vs CO2-physics-only retrains
│   ├── out_of_regime_sensor326.csv                # Failure-mode metrics
│   ├── performance_benchmark.csv                  # Loop vs NumPy timing, FLOPs, arithmetic intensity
│   ├── results_tables.md                          # All of the above as paste-ready Markdown
│   ├── run_metadata.txt                           # Library versions, machine, causal flag, selected model
│   └── pytest_report.txt                          # Output of `pytest -v` (see Section 7.2)
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
| **5. Bounded Gap Interpolation** | Linearly interpolate gaps up to `max_interp_gap_bins = 3` (`30 minutes`); leave longer outages as `NaN`. | `13,800` raw valid `co2_sensor_869` readings; `15,194` bins on the synchronized 10-min grid (`107` days x 144 = `15,408` possible). The difference is consistent with the bounded interpolation of outages up to 30 min and should be confirmed against the `Raw_data_processing.py` log | Avoids inventing synthetic diurnal cycles across multi-hour network outages. |

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

This relation implies that occupancy depends **jointly** on the instantaneous concentration excess `C(t) - C_amb` (steady-state ventilation balance) and the rate of change `dC/dt` (transient accumulation during arrivals and departures).

### 4.2 Finite-Difference Derivative & Physical Coordinate Contract
To evaluate `dC/dt` on discrete 10-minute measurements (`h = dt = 10.0 min`), interior points use the second-order centred finite-difference stencil with truncation error `O(h^2)`:

~~~text
dC/dt (at t_i) ≈ (C(t_{i+1}) - C(t_{i-1})) / (2 * dt) + O(dt^2)
~~~

while the left and right boundaries use first-order forward and backward stencils `O(dt)`. Differentiating with respect to the physical step size `dt = 10.0 min` (rather than array index `1`) is strictly enforced so derivatives carry physical units of `ppm/min`. To prevent high-frequency NDIR sensor quantization noise (`±5 ppm`) from amplifying derivative variance, a Savitzky-Golay smoothing filter (`window_length = 5`, `polyorder = 2`, `delta = 10.0 min`) is also evaluated.

**Causality of the model features (leakage control).** The centred stencil and SciPy's default Savitzky-Golay window both read **future** samples (`C(t + 10 min)`, `C(t + 20 min)`). Because CO2 lags occupancy by about 10 minutes (Section 5.2), a feature built from future CO2 would leak the present headcount into the model and could not exist in a real-time BMS. The model features `dcdt_raw` and `dcdt_savgol` therefore use **trailing** stencils (`compute_causal_co2_derivative`):

~~~text
dC/dt (at t_i) ≈ (3 C_i - 4 C_{i-1} + C_{i-2}) / (2 dt)      O(dt^2), i >= 2
dC/dt (at t_1) ≈ (C_1 - C_0) / dt                             O(dt),   i = 1
dC/dt (at t_0) = 0                                           (no history)
~~~

The Savitzky-Golay variant evaluates the local least-squares polynomial at the **newest** sample of the window (`scipy.signal.savgol_coeffs(..., pos = window_length - 1)`). Unit tests assert (i) second-order convergence of the causal stencil, (ii) exactness on a quadratic, and (iii) that perturbing samples at index `>= k` leaves the derivative at indices `< k` bit-for-bit unchanged, a property the centred filter fails. The centred stencils are retained only for offline analysis and for the numerical-verification tests. `run_pipeline(causal_features=False)` reproduces the earlier non-causal behaviour so the size of the leakage effect can be quantified.

---

## 5. Validation Design & Quantitative Results

### 5.1 Zero-Leakage Chronological Data Split (`co2_sensor_869`)
Because indoor `CO2` exhibits strong autocorrelation (`r > 0.90` at 10-minute lags), a random `train_test_split` leaks adjacent timestamps across training and test sets, producing artificially optimistic scores. We enforce a **strict chronological split at Melbourne midnight (`00:00:00`)** on the active primary sensor (`co2_sensor_869`), paired with **5-fold `TimeSeriesSplit` (`gap = 6` bins = 1-hour purge buffer)** on `Train + Validation`:

| Dataset Split | Melbourne Date Range | Calendar Days | 10-Min Rows | CO2 Range (`ppm`) | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set (70%)** | `2024-02-29` to `2024-05-12` | 74 days | `10,560` | `[403.0, 646.0]` | Parameter fitting & lag discovery |
| **Validation Set (15%)** | `2024-05-13` to `2024-05-28` | 16 days | `2,304` | `[420.0, 634.0]` | Validation reporting, feature-importance study & 5-fold CV pool (hyperparameters are fixed a priori, not searched) |
| **Withheld Test Set (15%)** | `2024-05-29` to `2024-06-14` | 17 days | `2,330` | `[411.0, 523.0]` | Final reporting only; model selection uses 5-fold CV RMSE, never test metrics |
| **Out-of-Regime (`326`)** | `2025-01-01` to `2026-03-30` (confirm: `31,342` bins is at least 218 days, not 212) | see note | `31,342` | `[400.0, 1387.0]` | Uncalibrated HVAC failure mode test |

**Operating range.** In this well-ventilated room CO2 never exceeds `646 ppm` in training and `523 ppm` in the test period. Occupancy signals are therefore small (a few hundred ppm at most), and any rule or model keyed to conventional absolute thresholds (e.g. `600 ppm`) is operating at the edge of, or outside, the observed range. The test period is also **distributionally shifted** relative to training: a mean-headcount predictor fitted on the training set scores a negative test `R2` (`-0.347`, Section 5.3), so every test `R2` below must be read against that reference.

### 5.2 Step 2 Results: Time-Lagged Cross-Correlation (`R(tau)`)
Cross-correlation on the Training Set between zone occupancy (`headcount_total_max`) and shifted `CO2` (`C(t + tau)`), `tau` in steps of 10 min:
- **Instantaneous correlation (`tau = 0 min`)**: `r = 0.6586`
- **Peak correlation (`tau* = +10 min`)**: `r = 0.6655`. CO2 lags occupancy by about one sample, but the gain over `tau = 0` is only `0.007`, so at 10-minute resolution the delay is barely resolvable (the lag cannot be estimated more finely than one bin).
- **Derivative (`dC/dt`, causal Savitzky-Golay)**: its correlation curve is plotted in `fig1_lagged_cross_correlation.png`; the peak value is not printed by the pipeline. Section 5.6 shows the feature adds no measurable predictive value.

### 5.3 Step 3 Results: Quantitative Comparison of Regression Models
All preprocessing (`SimpleImputer(strategy="median")` and `StandardScaler()`) is inside scikit-learn `Pipeline` objects so transformations are fitted **only on training data**. Physical clipping `N_hat(t) = max(0, y_pred(t))` is applied to every prediction. Models are selected on **5-fold `TimeSeriesSplit` CV RMSE** (Train + Validation); test metrics are reported for all models but never used for selection. Source: `results/model_comparison.csv` (Windows 11, Intel64 Family 6 Model 167, Python 3.14.4, scikit-learn 1.9.1; `results/run_metadata.txt`).

| Model | CV RMSE | Val MAE | Val RMSE | Val `R2` | Test MAE | Test RMSE | Test `R2` | Test MAE Empty (`N=0`) | Test MAE Occupied (`N>0`) | Fit (ms) | Predict, 2,330 rows (ms) | Fit peak (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Mean `DummyRegressor` | 9.799 | 9.468 | 13.187 | -0.097 | 4.745 | 5.544 | -0.347 | 8.852 | 4.688 | 7.2 | 0.007 | 0.003 |
| 0b. Engineering: `CO2 > 600 ppm` rule | 13.325 | 12.404 | 17.351 | -0.899 | 6.037 | 7.697 | -1.598 | 0.000 | 6.121 | 0.2 | 0.013 | 0.181 |
| 0c. Engineering: tuned CO2 threshold | 9.062 | 8.670 | 11.623 | 0.148 | 5.981 | 7.559 | -1.506 | 0.495 | 6.057 | 2.1 | 0.007 | 0.181 |
| 1. Static OLS | 7.109 | 6.932 | 9.274 | 0.457 | 3.558 | 4.750 | 0.011 | 3.458 | 3.559 | 2.0 | 0.246 | 0.436 |
| 2. Physical mass-balance OLS | 7.096 | 6.933 | 9.269 | 0.458 | 3.529 | 4.719 | 0.024 | 3.340 | 3.531 | 2.5 | 0.240 | 0.820 |
| 3. Ridge | 5.823 | 5.167 | 7.492 | 0.646 | 2.605 | 3.435 | 0.483 | 0.852 | 2.629 | 5.5 | 0.403 | 3.039 |
| 4. Decision Tree | 6.082 | 4.753 | 7.234 | 0.670 | 2.482 | 3.485 | 0.467 | 2.248 | 2.486 | 16.9 | 0.336 | 3.037 |
| **5. SVR (RBF)** (selected, lowest CV) | **5.732** | 4.673 | 7.215 | 0.672 | **2.169** | **3.114** | **0.575** | 1.511 | 2.178 | 2,111 | 701.3 | 3.039 |
| 6. MLP (32, 16) | 6.285 | 4.501 | 6.730 | 0.714 | 2.652 | 3.756 | 0.382 | 1.250 | 2.671 | 7,432 | 1.013 | 3.039 |

*Memory is peak Python/NumPy allocation during `fit` (`tracemalloc`). It does not capture C-level allocations such as the `libsvm` kernel cache used by SVR, so it understates SVR's true memory use. Predict time is the median of 10 repeats over the whole test set.*

#### Key Findings
1. **Gain over baselines.** The selected SVR cuts test RMSE from `5.544` (mean predictor) to `3.114` persons (**-43.8%**) and CV RMSE from `9.799` to `5.732` (**-41.5%**). Its test `R2` is `0.575`.
2. **The mass-balance term adds almost nothing.** Adding the causal `dC/dt` feature (Model 2 vs Model 1) changes test RMSE from `4.750` to `4.719` (**-0.7%**) and CV RMSE from `7.109` to `7.096`; the permutation importance of `dC/dt` is `-0.034` persons (Section 5.6). Once the feature is made causal (Section 4.2), the data do **not** support the claim that the transient term improves occupancy estimation at 10-minute resolution. A plausible, untested explanation is that a `+/-5 ppm` sensor resolution makes a backward difference over 10 minutes very noisy relative to the signal.
3. **Nonlinear multivariate models beat linear CO2-only models.** Ridge, Tree, SVR and MLP (all using the 8-feature set) reach test RMSE `3.1 to 3.8`, versus `4.7` for the CO2-only linear models. Section 5.6 shows that much of this gain comes from the time-of-day features rather than from CO2.
4. **Ridge is nearly as good as SVR and far cheaper.** SVR's CV RMSE advantage over Ridge is `1.6%` (`5.732` vs `5.823`), yet SVR takes about `387x` longer to fit and `1,740x` longer to predict. SVR is selected because it has the lowest CV RMSE (the pre-declared rule), but for resource-constrained deployment Ridge is the better engineering trade-off.
5. **Operating-region diagnostics.** Global metrics hide region effects: static OLS has a `3.458` person error when the room is empty, whereas Ridge reduces this to `0.852` and SVR to `1.511`. Decision-tree and MLP values are in the table.

### 5.4 Step 4 Results: Out-of-Regime Failure Mode Analysis (`co2_sensor_326`)
Models trained on `co2_sensor_869` (2024) were evaluated without refitting on `co2_sensor_326` (2025-2026). The CO2-occupancy correlation changes from `r = +0.6586` (training) to `r = -0.1897`. Source: `results/out_of_regime_sensor326.csv`.

| Model on `co2_sensor_326` | MAE (`persons`) | RMSE (`persons`) | `R2` | Reading |
| :--- | :---: | :---: | :---: | :--- |
| 0c. Tuned CO2 threshold | 6.246 | 9.134 | -0.925 | Fails: worse than predicting the mean. |
| 1. Static OLS | 5.828 | 8.490 | -0.663 | Fails. |
| 2. Physical mass-balance OLS | 5.802 | 8.420 | -0.636 | Fails: a fixed slope cannot hold when ventilation changes. |
| 4. Decision Tree | 3.864 | 5.814 | 0.220 | Retains some skill. |
| 5. SVR (RBF) | 4.835 | 6.540 | 0.013 | Approximately no skill beyond the mean. |

- **Conclusion.** The threshold rule and the linear CO2 models (negative `R2`) cannot be transferred to a different period with reconfigured ventilation without recalibration or airflow telemetry, which is a fundamental limit of single-zone CO2 sensing.
- **The nonlinear models fail less.** Because Section 5.6 shows these models depend heavily on time of day, a plausible explanation is that they keep partial skill by reproducing the occupancy *schedule*, not by sensing CO2. This was **not tested** on sensor 326 (no ablation was run there), so it is a hypothesis.

### 5.5 Engineering Baseline: Would a Simple CO2 Threshold Have Done as Well?
A `DummyRegressor` is only a statistical floor; the credible engineering alternative is the rule a facilities engineer would deploy: *the room is occupied when CO2 exceeds a threshold*. It is implemented as `CO2ThresholdRule` and evaluated on the same splits and metrics: **0b** uses a fixed `600 ppm` threshold, **0c** chooses the threshold on **training data only** (grid `420-1200 ppm`, step `10 ppm`). Source: `results/baseline_vs_selected.csv`.

| Model | CV RMSE | Test MAE | Test RMSE | Test `R2` | Test MAE Empty | Test MAE Occupied | Test RMSE vs tuned threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0. Mean `DummyRegressor` | 9.799 | 4.745 | 5.544 | -0.347 | 8.852 | 4.688 | -26.7% |
| 0b. `CO2 > 600 ppm` rule | 13.325 | 6.037 | 7.697 | -1.598 | 0.000 | 6.121 | +1.8% |
| 0c. Tuned CO2 threshold | 9.062 | 5.981 | 7.559 | -1.506 | 0.495 | 6.057 | 0% |
| 5. SVR (selected) | 5.732 | 2.169 | 3.114 | 0.575 | 1.511 | 2.178 | **-58.8%** |

**The threshold heuristic does not do as well; it is worse than predicting the mean.** The evidence:
- **The fixed rule never fires on the test set.** Test CO2 peaks at `523 ppm`, below `600 ppm`, so rule 0b predicts `0` persons for every sample (hence `Test MAE Empty = 0.000` and an occupied-hours MAE of `6.121`). The threshold is outside this room's operating range.
- **Tuning does not rescue it.** The tuned threshold (0c) achieves a positive validation `R2` (`0.148`) but a test `R2` of `-1.506`: a single absolute CO2 level learned in the training months does not transfer to the later test period, which has lower CO2 (peak `523 ppm`) and a shifted headcount distribution. A binary rule also cannot represent a headcount that varies between 0 and tens of persons.
- **Therefore the learned models add real value over the rule** (test RMSE `3.114` vs `7.559`, a `58.8%` reduction), but see Section 5.6 for what that value is made of.

*A stronger second baseline is still missing:* a pure **timetable** model (mean headcount for each hour of day) would test whether CO2 adds anything beyond the schedule. The "time-of-day only" feature set in Section 5.6 is the first step towards that comparison.

### 5.6 Feature Influence: Is the Model Sensing CO2 or Reading the Clock?
`sin_time`/`cos_time` and temperature/humidity are **not** terms of the mass-balance equation (Section 4.1). Two checks were run (`fig7_feature_importance_and_ablation.png`): (1) **grouped permutation importance** on the validation set (20 repeats, columns of a group are shuffled together so correlated features are not split), applied to the SVR; (2) **retraining ablation** of all four multivariate models on different feature sets.

**Grouped permutation importance, SVR** (unpermuted validation RMSE = `7.215` persons). Source: `results/feature_importance.csv`.

| Feature group | Features | RMSE increase when shuffled (persons, mean ± sd) |
| :--- | :--- | :---: |
| **Time of day** | `sin_time`, `cos_time` | **5.607 ± 0.072** |
| CO2 level & history | `co2_active`, `co2_lag1`, `co2_lag2` | 4.212 ± 0.101 |
| Temperature & humidity | `temp_c`, `humidity_rh` | 0.460 ± 0.062 |
| CO2 rate `dC/dt` | `dcdt_savgol` | -0.034 ± 0.021 |

**Retraining ablation, Test RMSE in persons (Test `R2`).** Source: `results/time_feature_ablation.csv`.

| Model | Full (8 features) | No time-of-day (6) | CO2 physics only (4) | Time of day only (2) |
| :--- | :---: | :---: | :---: | :---: |
| Ridge | 3.435 (0.483) | 4.339 (0.175) | 4.710 (0.027) | `TBD` |
| Decision Tree | 3.485 (0.467) | 4.381 (0.159) | 4.665 (0.046) | `TBD` |
| SVR (RBF) | 3.114 (0.575) | 4.357 (0.168) | 5.025 (-0.107) | `TBD` |
| MLP | 3.756 (0.382) | 4.387 (0.156) | 4.625 (0.062) | `TBD` |

*The "Time of day only" column was added after the run above; re-run the pipeline and copy it from `results/time_feature_ablation.csv`.*

**Interpretation.**
1. **The model is not purely sensing CO2.** Time of day is the most influential group, and removing it degrades every model: test RMSE rises by `17%` (MLP) to `40%` (SVR) and test `R2` falls from `0.38-0.58` to `0.16-0.18`. A substantial part of the skill is the building schedule, so the model should not be expected to transfer to a room or term with a different timetable (consistent with Section 5.4).
2. **CO2 does carry some signal, but weakly.** The CO2-physics-only models beat the mean predictor in cross-validation (CV RMSE `7.09-7.27` vs `9.80`, about `26-28%` lower) and reach validation `R2` of `0.43-0.47`, yet their test `R2` is `-0.11` to `0.06`. Part of this collapse is the train-to-test distribution shift noted in Section 5.1.
3. **The rate of change contributes nothing measurable** (importance indistinguishable from zero), and temperature/humidity contribute little (`0.46` persons) and are not physically justified here; they were retained for completeness.
4. **Limitation of the method.** Permutation importance can understate features whose information is duplicated elsewhere; that is why the retraining ablation is reported alongside it.

### 5.7 What the Validation Cannot Establish
- **One zone, one sensor, one semester.** The chronological test set is 17 days of a single room; it says nothing about other rooms or seasons (Section 5.4 shows transfer fails when ventilation changes).
- **Ground truth is itself a measurement.** Camera headcounts are aggregated with `.max()` per 10-minute bin and carry their own counting error, so the metrics measure agreement with the camera, not with true occupancy.
- **No hyperparameter search.** `alpha = 10`, `C = 10`, `epsilon = 1`, `max_depth = 5` and the MLP shape are fixed a priori; the reported ranking is therefore not tuned and could change with tuning.
- **Test metrics are reported for all models** and used only for reporting; selection is made on CV RMSE.
- **The test period is distributionally shifted** (mean-predictor test `R2 = -0.347`; CO2 maxima of `646 ppm` in training vs `523 ppm` in test), so absolute test `R2` values understate what the models do within a stable regime and overstate how hard the problem is across regimes. A rolling-origin evaluation would separate the two.
- **Much of the skill is schedule, not CO2** (Section 5.6), so the validation says little about performance in a room whose usage pattern differs.
- **The derivative term could not be validated** (Section 5.3, finding 2): the mass-balance model is a sound physical derivation, but at 10-minute resolution with sensor noise the data do not show it improving estimates.
- **Memory figures exclude C-level allocations** (SVR kernel cache).

---

## 6. Justification of Computational Approach & Performance Profiling

### 6.1 Algorithmic Complexity, Accuracy vs. Cost Trade-Off
Times are from a single run (Windows 11, Intel64 Family 6 Model 167, Python 3.14.4). `Predict` is the whole 2,330-row test set.

| Model | Complexity (training) | Fit | Predict (2,330 rows) | Test RMSE | Reading |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Static / ODE OLS | `O(N p^2 + p^3)` | 2.0-2.5 ms | 0.24 ms | 4.75 / 4.72 | Cheapest and interpretable, but little skill |
| Ridge | `O(N p^2 + p^3)` | 5.5 ms | 0.40 ms | 3.435 | Best accuracy per unit cost |
| Decision Tree (`max_depth=5`) | `O(N p log N)` | 16.9 ms | 0.34 ms | 3.485 | Interpretable thresholds; cannot extrapolate |
| **SVR (RBF, `C=10`, `epsilon=1`)** | `O(N^2 p)` to `O(N^3)` | 2,111 ms | 701 ms (0.30 ms/row) | **3.114** | Lowest CV and test error |
| MLP (32, 16) | `O(epochs N p h)` | 7,432 ms | 1.0 ms | 3.756 | Slowest to train, no accuracy gain |

- **Selection and justification.** SVR is the primary model because it has the lowest 5-fold CV RMSE (`5.732`), and its test RMSE is also the lowest. The margin over Ridge is small (`1.6%` in CV), so the choice is a trade-off: at `0.30 ms` per row SVR is still about `2 x 10^6` times faster than the 10-minute BMS sampling interval (`600,000 ms`), so runtime is not a practical obstacle on a building server; Ridge is the better option for a microcontroller.
- **What the accuracy does and does not mean.** The advantage over the mean and threshold baselines is large and robust (Section 5.5), but most of it comes from time-of-day features (Section 5.6).

### 6.2 Empirical Timing, FLOPS & Arithmetic Intensity Benchmark
Using `timeit.repeat(repeat=5, number=20)` in `benchmark_numerical_performance()`, a pure Python `for`-loop centred difference was compared with vectorized `numpy.gradient` on the `N = 10,560` float64 training array. Source: `results/performance_benchmark.csv`.

| Implementation | `N` | Mean runtime | Std dev | Speedup | FLOPs (`2N`) | Memory traffic (`16N` B) | Arithmetic intensity | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Python loop | 10,560 | 2.587 ms | 0.114 ms | 1.0x | 21,120 | 168,960 B | 0.125 FLOP/B | 8.2 MFLOP/s |
| `numpy.gradient` | 10,560 | **0.0192 ms** | 0.0013 ms | **135.1x** | 21,120 | 168,960 B | 0.125 FLOP/B | **1,102.7 MFLOP/s** |

- **Interpretation.** Each point costs about 2 FLOPs (one subtraction, one division) and moves about 16 bytes, an arithmetic intensity of `0.125 FLOP/byte` (an *assumed* traffic model, not a hardware-counter measurement). Such a low intensity places the operation on the memory-bound side of any roofline, so the speed gain is attributable to removing Python interpreter overhead (type checks, loop dispatch) in favour of a compiled loop, not to higher arithmetic throughput. The `1.1 GFLOP/s` is far below a modern CPU's peak, consistent with a memory-bound kernel. No hardware profiler was used; the figures are `timeit` wall-clock measurements.
- **Relevance.** The derivative is not the pipeline's cost bottleneck (SVR fitting is, at about `2.1 s`); the benchmark demonstrates the method for a representative numerical kernel and is not the main optimisation result.

---

## 7. Automated Verification & Unit Testing (`pytest`)

The automated test suite in `tests/test_pipeline.py` contains **24 unit tests** (14 positive verification tests and 10 negative exception-handling tests) using `pytest.approx()` for floating-point round-off tolerance and `pytest.raises(ValueError)` for defensive input rejection:

| Test Function | Category | Verification Target & Acceptance Criterion | Status (`results/pytest_report.txt`) |
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
| `test_positive_causal_derivative_second_order_convergence` | Positive (Convergence) | Causal backward stencil: `p_obs == 2.0` when `h` is halved twice (`rel=5e-2`). | `run to confirm` |
| `test_positive_causal_savgol_exact_on_quadratic` | Positive (Analytical) | Trailing Savitzky-Golay derivative of `3 + 2t + 0.05t^2` equals `2 + 0.1t` once a full window exists (`rel=1e-9`). | `run to confirm` |
| `test_positive_causal_derivative_ignores_future_samples` | Positive (Leakage guard) | Perturbing samples `>= k` leaves the causal derivative at indices `< k` unchanged; the centred Savitzky-Golay filter fails the same check. | `run to confirm` |
| `test_positive_threshold_rule_known_values` | Positive (Baseline) | `CO2ThresholdRule(600)` returns `0` below and the hand-computed mean headcount (`15`) above the threshold. | `run to confirm` |
| `test_positive_tuned_threshold_recovers_step_change` | Positive (Baseline) | On an exact 450/800 ppm step the tuned threshold lies in `[450, 800)` and training MAE is `0`. | `run to confirm` |
| `test_positive_grouped_permutation_importance_identifies_driver` | Positive (Diagnostics) | For `y = 5*x0 + 40`, shuffling `x0` raises RMSE by `> 1` while shuffling the irrelevant `x1` changes it by `~0`. | `run to confirm` |
| `test_negative_causal_derivative_invalid_inputs` | Negative (`ValueError`) | Rejects `dt <= 0`, `NaN`, `< 3` samples, and an even/short Savitzky-Golay window. | `run to confirm` |
| `test_negative_threshold_rule_rejects_nan_and_multicolumn_input` | Negative (`ValueError`) | Rejects `NaN` CO2, more than one input column, and mismatched target length. | `run to confirm` |
| `test_negative_grouped_permutation_importance_invalid_arguments` | Negative (`ValueError`) | Rejects an unknown feature name, an empty group, and inconsistent shapes. | `run to confirm` |

### 7.1 Measured Order of Accuracy
Computed by `observed_convergence_order()` on `sin(omega t)`, `period = 120 min`, `h = 2.0 -> 1.0 -> 0.5 min` (maximum interior error, deterministic):

| Stencil | `p_obs` (2.0 -> 1.0) | `p_obs` (1.0 -> 0.5) | Expected |
| :--- | :---: | :---: | :---: |
| Centred, `O(h^2)` | `1.9994` | `1.9999` | `2` |
| Causal backward, `O(h^2)` | `1.9988` | `1.9997` | `2` |

### 7.2 Test Report
Generate and commit the report so the outcome (not only the specification) is on record:

~~~bash
python -m pytest -v --junitxml=results/pytest_report.xml 2>&1 | tee results/pytest_report.txt
~~~

Paste the final summary line here after running (expected: `24 passed`):

~~~text
TBD  <- paste the last line of results/pytest_report.txt
~~~

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
This prints the derivative-stencil verification, chronological split summary, lagged correlation results, the 9-model comparison table, the baseline-vs-selected comparison, grouped permutation importance and time-feature ablation, the Sensor 326 failure analysis, and the FLOPS benchmark. It saves all 16 figures (300 DPI) into `Figures/` and every table as CSV / paste-ready Markdown into `results/`.

To quantify how much future-sample leakage would have inflated the scores, call `run_pipeline(output_dir="Figures_noncausal", results_dir="results_noncausal", causal_features=False)` from a Python shell and compare the two `model_comparison.csv` files.

### Step 4: Run the Automated `pytest` Verification Suite
~~~bash
python -m pytest -v --junitxml=results/pytest_report.xml 2>&1 | tee results/pytest_report.txt
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