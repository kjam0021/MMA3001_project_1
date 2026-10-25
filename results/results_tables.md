### Model comparison (all models, chronological split)

| Model | CV_RMSE | Val_MAE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 | Test_MAE_Empty | Test_MAE_Occ | Fit_ms | Pred_ms_testset | Fit_peak_MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0. Baseline: Mean DummyRegressor | 9.799 | 9.468 | 13.187 | -0.097 | 4.745 | 5.544 | -0.347 | 8.852 | 4.688 | 6.210 | 0.007 | 0.003 |
| 0b. Engineering Baseline: CO2 > 600 ppm rule | 13.325 | 12.404 | 17.351 | -0.899 | 6.037 | 7.697 | -1.598 | 0.000 | 6.121 | 0.171 | 0.007 | 0.181 |
| 0c. Engineering Baseline: CO2 threshold tuned on train | 9.062 | 8.670 | 11.623 | 0.148 | 5.981 | 7.559 | -1.506 | 0.495 | 6.057 | 2.065 | 0.007 | 0.181 |
| 1. Linear Regression: Static OLS | 7.109 | 6.932 | 9.274 | 0.457 | 3.558 | 4.750 | 0.011 | 3.458 | 3.559 | 2.006 | 0.239 | 0.436 |
| 2. Physical Mass-Balance ODE OLS | 7.096 | 6.933 | 9.269 | 0.458 | 3.529 | 4.719 | 0.024 | 3.340 | 3.531 | 2.343 | 0.247 | 0.820 |
| 3. Regularized Ridge Regression | 5.823 | 5.167 | 7.492 | 0.646 | 2.605 | 3.435 | 0.483 | 0.852 | 2.629 | 5.816 | 0.407 | 3.039 |
| 4. Decision Tree Regressor | 6.082 | 4.753 | 7.234 | 0.670 | 2.482 | 3.485 | 0.467 | 2.248 | 2.486 | 16.590 | 0.352 | 3.037 |
| 5. Support Vector Regression (RBF) | 5.732 | 4.673 | 7.215 | 0.672 | 2.169 | 3.114 | 0.575 | 1.511 | 2.178 | 2160.755 | 723.856 | 3.039 |
| 6. Neural Network (MLPRegressor) | 6.285 | 4.501 | 6.730 | 0.714 | 2.652 | 3.756 | 0.382 | 1.250 | 2.671 | 6877.839 | 0.763 | 3.039 |

### Selected model vs engineering baseline

| Model | CV_RMSE | Test_MAE | Test_RMSE | Test_R2 | Test_MAE_Empty | Test_MAE_Occ | Fit_ms | Pred_ms_testset | Fit_peak_MB | Test_RMSE_change_vs_tuned_threshold_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0. Baseline: Mean DummyRegressor | 9.799 | 4.745 | 5.544 | -0.347 | 8.852 | 4.688 | 6.210 | 0.007 | 0.003 | -26.665 |
| 0b. Engineering Baseline: CO2 > 600 ppm rule | 13.325 | 6.037 | 7.697 | -1.598 | 0.000 | 6.121 | 0.171 | 0.007 | 0.181 | 1.827 |
| 0c. Engineering Baseline: CO2 threshold tuned on train | 9.062 | 5.981 | 7.559 | -1.506 | 0.495 | 6.057 | 2.065 | 0.007 | 0.181 | 0.000 |
| 5. Support Vector Regression (RBF) | 5.732 | 2.169 | 3.114 | 0.575 | 1.511 | 2.178 | 2160.755 | 723.856 | 3.039 | -58.807 |

### Grouped permutation importance - 5. Support Vector Regression (RBF) (validation set, RMSE increase in persons)

| Feature_Group | Features | RMSE_increase_mean | RMSE_increase_std |
| --- | --- | --- | --- |
| Time of day (sin/cos) | sin_time, cos_time | 5.607 | 0.072 |
| CO2 level & history | co2_active, co2_lag1, co2_lag2 | 4.212 | 0.101 |
| Temperature & humidity | temp_c, humidity_rh | 0.460 | 0.062 |
| CO2 rate dC/dt | dcdt_savgol | -0.034 | 0.021 |

### Feature-set ablation (does the model read the clock or sense CO2?)

| Model | Feature_Set | N_features | CV_RMSE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3. Regularized Ridge Regression | Full (8 features) | 8 | 5.823 | 7.492 | 0.646 | 2.605 | 3.435 | 0.483 |
| 3. Regularized Ridge Regression | No time-of-day | 6 | 6.588 | 8.686 | 0.524 | 3.270 | 4.338 | 0.175 |
| 3. Regularized Ridge Regression | CO2 physics only | 4 | 7.090 | 9.269 | 0.458 | 3.508 | 4.710 | 0.027 |
| 3. Regularized Ridge Regression | Time of day only | 2 | 7.347 | 10.245 | 0.338 | 4.021 | 5.320 | -0.241 |
| 4. Decision Tree Regressor | Full (8 features) | 8 | 6.082 | 7.234 | 0.670 | 2.482 | 3.485 | 0.467 |
| 4. Decision Tree Regressor | No time-of-day | 6 | 6.932 | 8.417 | 0.553 | 3.237 | 4.381 | 0.159 |
| 4. Decision Tree Regressor | CO2 physics only | 4 | 7.268 | 9.300 | 0.454 | 3.466 | 4.665 | 0.046 |
| 4. Decision Tree Regressor | Time of day only | 2 | 7.246 | 10.065 | 0.361 | 3.905 | 5.337 | -0.249 |
| 5. Support Vector Regression (RBF) | Full (8 features) | 8 | 5.732 | 7.215 | 0.672 | 2.169 | 3.114 | 0.575 |
| 5. Support Vector Regression (RBF) | No time-of-day | 6 | 6.619 | 8.448 | 0.550 | 3.108 | 4.357 | 0.168 |
| 5. Support Vector Regression (RBF) | CO2 physics only | 4 | 7.229 | 9.485 | 0.433 | 3.577 | 5.025 | -0.107 |
| 5. Support Vector Regression (RBF) | Time of day only | 2 | 7.363 | 10.660 | 0.283 | 3.303 | 4.521 | 0.104 |
| 6. Neural Network (MLPRegressor) | Full (8 features) | 8 | 6.285 | 6.730 | 0.714 | 2.652 | 3.756 | 0.382 |
| 6. Neural Network (MLPRegressor) | No time-of-day | 6 | 6.916 | 8.095 | 0.587 | 3.114 | 4.387 | 0.156 |
| 6. Neural Network (MLPRegressor) | CO2 physics only | 4 | 7.167 | 9.153 | 0.472 | 3.424 | 4.625 | 0.062 |
| 6. Neural Network (MLPRegressor) | Time of day only | 2 | 7.263 | 10.067 | 0.361 | 3.886 | 5.317 | -0.239 |

### Out-of-regime failure analysis (sensor 326)

| Model | Sensor326_MAE | Sensor326_RMSE | Sensor326_R2 |
| --- | --- | --- | --- |
| 0c. Engineering Baseline: CO2 threshold tuned on train | 6.246 | 9.134 | -0.925 |
| 1. Linear Regression: Static OLS | 5.828 | 8.490 | -0.663 |
| 2. Physical Mass-Balance ODE OLS | 5.802 | 8.420 | -0.636 |
| 4. Decision Tree Regressor | 3.864 | 5.814 | 0.220 |
| 5. Support Vector Regression (RBF) | 4.835 | 6.540 | 0.013 |

### Derivative benchmark (Python loop vs NumPy)

| N_samples | Python_Loop_ms | Python_Loop_std_ms | NumPy_Vectorized_ms | NumPy_Vectorized_std_ms | Speedup_Factor | Arithmetic_Intensity_FLOP_per_Byte | Vectorized_MFLOPS_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10560.000 | 2.845 | 0.323 | 0.020 | 0.002 | 144.138 | 0.125 | 1070.071 |

