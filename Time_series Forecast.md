Got it ✅ — we’ll focus **only on revenue metrics** (ADR, RevPAR, Occupancy %, and Revenue) **without any segments**. Here’s a fully explicit IDE prompt for that:

---

**Prompt:**

\*"I have daily hotel data for 2022–2025 with the following columns: Date, Revenue, Occupancy, ADR, and RevPAR. Segment information is in a separate report and should be ignored.

Please perform the following steps explicitly:

1. **Load all CSV files** for the four years and combine them into a single dataset.
2. **Parse the Date column** as datetime and sort the data in ascending order.
3. **Check for missing dates** and fill them:

   * Forward-fill Occupancy, ADR, RevPAR, and Revenue for missing dates
   * If Revenue is missing, set it to 0
4. **Validate and calculate metrics** if any are missing or inconsistent:

   * ADR = Revenue / Rooms Sold (if Rooms Sold is available)
   * RevPAR = ADR \* Occupancy %
5. **Ensure all columns are in proper numeric format**: Revenue, Occupancy, ADR, RevPAR.
6. **Visualize the data** to inspect trends, seasonality, and anomalies:

   * Daily, weekly, and monthly Revenue
   * Daily, weekly, and monthly ADR
   * Daily, weekly, and monthly RevPAR
   * Daily, weekly, and monthly Occupancy %
7. **Perform correlation analysis** among Revenue, ADR, RevPAR, and Occupancy to understand relationships.
8. **Prepare the dataset** for time series forecasting or descriptive analytics.
9. **Export the cleaned and prepared dataset** to a new CSV file.
10. Make the code modular, well-commented, and production-ready so it can be reused for other hotel datasets with similar metrics."\*

---

This is **purely metric-focused** and **does not involve segments at all**.

If you want, I can also make a **follow-up prompt specifically for forecasting Revenue using only these metrics**, keeping ADR, RevPAR, and Occupancy as optional regressors. Do you want me to do that next?

1. **Data & Ingestion** — Combine all yearly CSVs into one time-indexed table; parse `Date` to datetime, sort ascending, check/fill missing dates (Revenue→0 or interpolate; forward-fill Occupancy/ADR/RevPAR). Ensure numeric types and consistent time zone.
2. **Exploratory Data Analysis (EDA)** — inspect seasonality, outliers, and holidays/events; run ADF/KPSS for stationarity and show ACF/PACF.
3. **Preprocessing & Transformations** — Apply log or Box-Cox if variance-stabilizing; difference if nonstationary; create calendar features (day-of-week, month, is\_holiday), rolling statistics (lags, rolling mean/std), and future regressor placeholders (e.g., expected Occupancy). Scale features only for ML/DL models.
4. **Feature Engineering (for multivariate/ML models)** — Build lag features (t-1, t-7, t-14), lead/lag for Occupancy/ADR/RevPAR, rolling windows (7/30/90 days), and indicator features for special events. Keep a clear feature dictionary and timestamp alignment.
5. **Model Candidates (list)** —

   * **Classical univariate**: ARIMA / SARIMA, Holt-Winters (ETS), Exponential Smoothing.
   * **Regression with time series**: SARIMAX (exogenous regressors), Dynamic Linear Models / State-Space.
   * **Multiple-series / multivariate**: VAR for multiple continuous series.
   * **Robust seasonal**: TBATS (multiple/complex seasonality).
   * **Interpretable additive**: Prophet / NeuralProphet.
   * **Tree-based ML**: XGBoost / LightGBM / RandomForest on lagged features.
   * **Deep learning**: LSTM, GRU, Temporal Fusion Transformer (TFT), Transformers for time series.
   * **Hybrid & ensembles**: combine top models (stacking/weighted average).
6. **Training Strategy & Validation** — Use **time-aware splits**: hold out a contiguous test period (e.g., last 90 days) and use **rolling/expanding window cross-validation** (walk-forward/backtesting). For each fold, retrain and forecast the horizon; record metrics. Do hyperparameter tuning within each fold (grid/random/Optuna).
7. **Evaluation Metrics & Uncertainty** — Report RMSE, MAE, MAPE, sMAPE, and (for probabilistic models) Prediction Interval Coverage and Pinball Loss. Compare point forecasts and interval calibration. Use relative improvement vs. a naive seasonal-naive baseline.
8. **Diagnostics & Residual Checks** — Check residual ACF/PACF, Ljung-Box test for autocorrelation, normality of residuals, and heteroscedasticity. Revisit differencing, transformation, or add regressors if diagnostics fail.
9. **Model Selection, Ensembling & Explainability** — Rank models by validation metrics and stability across folds. Build a simple ensemble (weighted average or stacking) if it improves validation. For ML models, use SHAP or partial dependence to explain feature importance (lags, occupancy, ADR effects).
10. **Productionization & Deliverables** — Save final model(s) and preprocessing pipeline; produce: (a) CSV of historical + forecast + lower/upper intervals, (b) evaluation report and plots, (c) backtest results, and (d) deployment plan (serialize model, expose REST endpoint, scheduled retraining). Include clear instructions on how to supply future regressor scenarios (e.g., assumed occupancy path) when generating forecasts.

---

If you want, I’ll convert point **10** into a single **IDE prompt** (no code) that instructs an assistant to implement the entire pipeline end-to-end (data prep → models listed above → rolling CV → diagnostics → export & deployment). Which would you prefer — the plain prompt now, or directly formatted for Prophet/ARIMA/ML separately?

1. **Data & Ingestion** — Combine all yearly CSVs into one time-indexed table; parse `Date` to datetime, sort ascending, check/fill missing dates (Revenue→0 or interpolate; forward-fill Occupancy/ADR/RevPAR). Ensure numeric types and consistent time zone.
2. **Exploratory Data Analysis (EDA)** — Plot daily/weekly/monthly Revenue, ADR, RevPAR, Occupancy; inspect trend, seasonality, outliers, and holidays/events; run ADF/KPSS for stationarity and show ACF/PACF.
3. **Preprocessing & Transformations** — Apply log or Box-Cox if variance-stabilizing; difference if nonstationary; create calendar features (day-of-week, month, is\_holiday), rolling statistics (lags, rolling mean/std), and future regressor placeholders (e.g., expected Occupancy). Scale features only for ML/DL models.
4. **Feature Engineering (for multivariate/ML models)** — Build lag features (t-1, t-7, t-14), lead/lag for Occupancy/ADR/RevPAR, rolling windows (7/30/90 days), and indicator features for special events. Keep a clear feature dictionary and timestamp alignment.
5. **Model Candidates (list)** —

   * **Classical univariate**: ARIMA / SARIMA, Holt-Winters (ETS), Exponential Smoothing.
   * **Regression with time series**: SARIMAX (exogenous regressors), Dynamic Linear Models / State-Space.
   * **Multiple-series / multivariate**: VAR for multiple continuous series.
   * **Robust seasonal**: TBATS (multiple/complex seasonality).
   * **Interpretable additive**: Prophet / NeuralProphet.
   * **Tree-based ML**: XGBoost / LightGBM / RandomForest on lagged features.
   * **Deep learning**: LSTM, GRU, Temporal Fusion Transformer (TFT), Transformers for time series.
   * **Hybrid & ensembles**: combine top models (stacking/weighted average).
6. **Training Strategy & Validation** — Use **time-aware splits**: hold out a contiguous test period (e.g., last 90 days) and use **rolling/expanding window cross-validation** (walk-forward/backtesting). For each fold, retrain and forecast the horizon; record metrics. Do hyperparameter tuning within each fold (grid/random/Optuna).
7. **Evaluation Metrics & Uncertainty** — Report RMSE, MAE, MAPE, sMAPE, and (for probabilistic models) Prediction Interval Coverage and Pinball Loss. Compare point forecasts and interval calibration. Use relative improvement vs. a naive seasonal-naive baseline.
8. **Diagnostics & Residual Checks** — Check residual ACF/PACF, Ljung-Box test for autocorrelation, normality of residuals, and heteroscedasticity. Revisit differencing, transformation, or add regressors if diagnostics fail.
9. **Model Selection, Ensembling & Explainability** — Rank models by validation metrics and stability across folds. Build a simple ensemble (weighted average or stacking) if it improves validation. For ML models, use SHAP or partial dependence to explain feature importance (lags, occupancy, ADR effects).
10. **Productionization & Deliverables** — Save final model(s) and preprocessing pipeline; produce: (a) CSV of historical + forecast + lower/upper intervals, (b) evaluation report and plots, (c) backtest results, and (d) deployment plan (serialize model, expose REST endpoint, scheduled retraining). Include clear instructions on how to supply future regressor scenarios (e.g., assumed occupancy path) when generating forecasts.

---

If you want, I’ll convert point **10** into a single **IDE prompt** (no code) that instructs an assistant to implement the entire pipeline end-to-end (data prep → models listed above → rolling CV → diagnostics → export & deployment). Which would you prefer — the plain prompt now, or directly formatted for Prophet/ARIMA/ML separately?

1. **Recommendation (what to choose)** — Pick the horizon that matches your business need. For hotels I recommend  **two concurrent horizons** : a 90-day (≈3 months) operational forecast and a 12-month strategic forecast.
2. **Short-term (1 day → 3 months)** — Use for rate decisions, day-to-day yield management and tactical staffing. High accuracy achievable; models can leverage recent patterns and occupancy as regressors.
3. **Tactical / Medium (3 → 6 months)** — Good for promotions, channel planning and quarterly budgeting. Accuracy drops vs short-term but still useful for planning.
4. **Strategic / Long-term (6 → 12+ months)** — Use for annual budgeting, capital planning, and scenario analysis. Uncertainty grows; present wide prediction intervals.
5. **Booking window & seasonality (hotel rule)** — Your forecast horizon should at least cover the typical booking lead time and major season(s). If bookings come 120 days in advance, include ≥4-month forecasts.
6. **Data sufficiency & uncertainty** — With four years of daily data you can reasonably train models for 12-month forecasts, but expect higher error and larger intervals as horizon increases.
7. **Model choice by horizon** — Short: SARIMA/SARIMAX or Prophet (with occupancy as regressors). Medium: tree-based models (XGBoost/LightGBM) on lag features or Prophet hybrids. Long: TBATS, TFT/transformer or ensembles. Always include a seasonal-naive baseline.
8. **Validation approach** — Use time-aware validation (rolling/expanding window backtest). Validate each horizon separately (e.g., evaluate 30-, 90-, 180-, 365-day forecast errors) and report RMSE, MAE, MAPE and interval coverage.
9. **Operational best practice** — Produce a **daily/weekly rolling 90-day forecast** for operations and a monthly refreshed 12-month scenario forecast for strategy. Provide upper/lower prediction intervals and scenario inputs (assumed occupancy, events).
10. **Concrete final setup (practical)** — Start with **90-day** as your default operational prediction and **12-month** for strategic planning. Backtest both (last 90 and last 365 days), compare against seasonal-naive, then decide if you need a 6-month middle ground.
