# Natural Gas Storage Forecasting & Trading Strategy

A quantitative pipeline that forecasts weekly EIA natural gas storage changes from weather and production fundamentals, then backtests a futures trading strategy around the Thursday EIA report release.

---

## The Idea

Every Thursday at 10:30 AM ET, the EIA releases weekly natural gas storage data. Markets move sharply on the surprise — the gap between actual and consensus expectations. If storage changes are predictable from physical fundamentals (weather, inventory levels, production shocks), that forecast can generate edge relative to a seasonal baseline.

**Core equation:** `ΔStorage = Production − Consumption − Exports`

Production and exports are relatively stable week-to-week. Consumption is driven by temperature. This makes storage changes largely forecastable from weather — which is the entire predictive thesis.

---

## Data Sources

| Source | Data | API |
|--------|------|-----|
| EIA | Weekly working gas storage (2010–2026) | EIA Open Data API |
| NOAA | Daily temperatures — 5 demand cities (NYC, Chicago, Houston, Denver, Atlanta) | NOAA CDO API |
| NOAA | Daily min temperatures — 4 production basins (Permian, Marcellus, Haynesville, DJ Basin) | NOAA CDO API |
| Yahoo Finance | Henry Hub futures (NG=F) daily/hourly prices | yfinance |

---

## Feature Engineering

| Feature | Physical Justification |
|---------|----------------------|
| **HDD** (Heating Degree Days) | Days below 65°F, summed weekly. Linearizes the V-shaped demand-temperature relationship for heating demand. |
| **CDD** (Cooling Degree Days) | Days above 65°F, summed weekly. Captures summer power burn for air conditioning. |
| **% Storage Full (lag 1)** | Last week's storage level as % of capacity. One of the most load-bearing features in the model — removing it costs ~14% CV MSE, the single largest feature-removal impact measured. Note: storage in this 2010–2026 sample never goes below ~19% or above ~95% full, so a Darcy-flow-style near-capacity injectivity effect was tested (a "storage vs. calendar-week seasonal norm" feature) and did not hold up empirically — see Limitations. |
| **Storage Change (lag 1)** | Last week's storage change. Autoregressive momentum term. |
| **Freeze Intensity** | Production-weighted sum of degrees below 15°F across Marcellus, Permian, Haynesville, and DJ Basin production regions. Below 15°F, hydrate formation in wellhead equipment causes supply disruptions independent of demand — a ChemE-motivated feature that simultaneously shocks supply and demand. |
| **HDD / CDD Hinge Terms** | `max(x − τ, 0)`, τ = the 75th-percentile HDD/CDD value computed from each CV fold's training data only. A piecewise-linear (Hansen-style threshold regression) term that lets extreme-cold and extreme-hot weeks pick up their own slope instead of sharing one line with ordinary weeks. Validated via a τ-sensitivity sweep showing a genuine V-shaped error minimum, not a monotonic artifact of curve-fitting. |
| **Holiday Week** | Flags weeks containing a US federal holiday. Commercial/industrial demand drops when offices are closed, independent of weather — the gain comes mostly from cleaning up the shared weather coefficients for the other ~95% of weeks, not from fixing holiday weeks' own error directly. |

**Key design choice:** Weather features cover Monday–Thursday of the EIA report week (data available before Thursday 10:30 AM release). Storage lag features use the prior week's released data. No look-ahead bias.

---

## Models

Evaluated with `sklearn.TimeSeriesSplit` (5-fold walk-forward CV): each fold trains only on the past and tests only on the future, never randomly shuffled. Any training-derived statistic (e.g. the hinge τ) is recomputed from that fold's training data only, to avoid leakage.

| Model | CV R² | CV MSE |
|-------|-------|--------|
| HDD + CDD (baseline) | 0.934 | 641.6 |
| + % Storage Full | 0.936 | 622.7 |
| + Storage Change Lag | 0.943 | 556.6 |
| + Freeze Event (binary) | 0.942 | 560.4 |
| + Freeze Intensity (weighted) | 0.947 | 513.5 |
| XGBoost (all features, no residual staging) | 0.946 | 520.7 |
| + HDD/CDD Hinge terms (linear only) | 0.949 | 488.6 |
| Linear + XGBoost-on-residuals (two-stage) | 0.951 | 467.4 |
| Hinge + XGBoost-on-residuals | 0.952 | 453.1 |
| **+ Holiday Week (final / champion)** | **0.953** | **441.2** |

**Champion architecture — a two-stage estimator, not a single model:**
```
Stage 1 (OLS):      ŷ₁ = β₀ + β₁·HDD + β₂·CDD + β₃·pct_full_lag1 + β₄·storage_change_lag1
                          + β₅·freeze_intensity + β₆·holiday_week
                          + β₇·max(HDD−τ_hdd, 0) + β₈·max(CDD−τ_cdd, 0)
Stage 2 (XGBoost):  trained on Stage 1's training-fold residuals only (200 trees, depth 3,
                    lr 0.05, subsample 0.8, min_child_weight 5, seeded — see Limitations)
Final:              ŷ = ŷ₁ + XGB_correction(X)
```
Stage 1 captures the strong, genuinely linear seasonal signal efficiently (a tree ensemble is a poor way to represent a straight line); Stage 2 only has to explain whatever nonlinear structure is left over.

**Key finding — heteroscedasticity, not just fit quality:** residual variance is not constant. On the ~5% most severe freeze weeks (`freeze_intensity` ≥ 95th percentile, ~44 weeks across 16 years), residual variance runs ~2.1× that of ordinary weeks (RMSE ≈ 31.7 BCF vs. ≈ 20.2 BCF), a direct violation of the OLS homoscedasticity assumption. Pooled R² stays high (95.3%) despite this because total variance in `storage_change` is dominated by the easily-explained seasonal swing — weather alone already explains 93.4% of it — so the heteroscedastic tail barely moves a pooled metric even though it's the highest-stakes segment of the forecast.

**Fourteen additional features were tested to close that tail gap; four survived, ten were rejected** — including a feature interaction that turned out 0.99-correlated with an existing one, a squared term that blew up under CV due to leverage from a single extreme week, a rare-event dummy too sparse to estimate reliably, and a raw linear time trend that failed catastrophically by extrapolating outside its training range under walk-forward CV. The consistent failure of unrelated fixes aimed at the same tail is itself evidence: this reads as a genuine sample-size ceiling (~44 severe-freeze weeks in 16 years) rather than a fixable specification problem.

---

## Trading Strategy

**Setup:**
- Signal: model prediction minus 5-year seasonal average for that week
- Trade when signal exceeds 1σ of historical benchmark error (high conviction only)
- Entry: Wednesday 11 PM close | Exit: Thursday 10 AM (post-EIA release)
- Cost: $0.01/MMBtu per contract, 10,000 MMBtu contract size

**Full strategy results (2015–2026):**

| Metric | Value |
|--------|-------|
| Total P&L | -$25,390 |
| Win rate | 53.4% |
| Sharpe (annualized) | -0.49 |

**Short-only results:**

| Metric | Value |
|--------|-------|
| Trades | 38 |
| Win rate | 65.8% (25/38) |
| Sharpe (annualized) | 1.45 |
| Total P&L | +$8,950 |
| p-value (vs. random) | 0.037 |
| Statistically significant | Yes (5% level) |

**Why short-only works, long-only doesn't:**

Short trades are triggered when the model predicts more supply than the seasonal baseline — a bearish surprise. These convert to price moves reliably because injection-season surpluses are driven by stable, predictable weather patterns.

Long trades (bullish surprise — predicting larger-than-expected withdrawals) fail despite correct directional forecasts. Natural gas prices in winter already price in cold weather expectations by Wednesday. When the bullish storage number releases Thursday, the market "sells the news." This asymmetry is a real market microstructure finding, not a data artifact.

---

## Limitations

1. **Sample size:** 38 short trades over 10 years. Statistically significant but a larger sample (500+) would be needed before live deployment.
2. **Execution assumptions:** Perfect fills assumed. Real bid-ask spread and slippage near the EIA release (thin liquidity) would reduce P&L by an estimated 30–50%.
3. **Benchmark validity:** 5-year seasonal average proxies market consensus. If professional forecasters already use this benchmark, the edge may be partially priced in.
4. **Regime stability:** Model trained on 2010–2026. LNG export growth post-2016 and AI data center demand post-2023 represent structural shifts not fully captured.
5. **Freeze intensity proxy:** Production-basin TMIN is a weather proxy for supply disruptions. Direct pipeline flow data (proprietary) would be more precise.
6. **No Darcy-flow injectivity effect observed:** storage level never approaches true physical extremes in this sample (min ~19% full, max ~95% full) — a "storage level relative to its own calendar-week historical norm" feature was tested (the industry's "storage vs. 5-year average" concept) and did not improve the model; the raw `pct_full_lag1` level outperformed every seasonally-adjusted variant tried.
7. **Point predictions only, no formal interval:** the model currently outputs a single number per week, not a calibrated error range. A regime-conditional empirical-quantile band (using the champion's out-of-fold residuals, split by `freeze_severe`) was prototyped but found too coarse — every week in a bucket got the same wide band regardless of its actual severity. A continuous version (scaling the band with `freeze_intensity` directly via linear quantile regression) is the planned next step.
8. **Reproducibility:** `XGBRegressor`'s `subsample=0.8` is stochastic; all instances are now seeded (`random_state=42`) so results are identical across runs — this was not always true and was caught and fixed mid-project.
9. **Partial-week data risk:** the weekly weather aggregation (`resample("W-THU")`) does not check whether a trailing week has a full 7 days of underlying daily data before summing it, so the most recent week in `weekly_degree_days.csv`/`weekly_production_tmin.csv` can silently understate HDD/CDD/freeze_intensity if NOAA's reporting lag hasn't caught up yet. Currently harmless only because EIA's storage data itself lags behind that partial week — a guard against this has not yet been added.

---

## File Structure

```
├── features.py                  # HDD/CDD computation from temperature series
├── weather.py                   # NOAA demand-city weather pipeline
├── production_weather.py        # NOAA production-basin freeze intensity
├── data.py                      # EIA storage data pipeline
├── model.py                     # Feature engineering + model comparison
├── backtest.py                  # Walk-forward trading backtest
├── intraday_backtest.py         # Intraday execution (Wed 11PM → Thu 10AM)
└── README.md
```

---

## How to Run

```bash
# Set up environment variables
echo "EIA_API_KEY=your_key" > .env
echo "NOAA_TOKEN=your_token" >> .env

# Pull data (takes ~45 min total for NOAA)
python data.py
python weather.py
python production_weather.py

# Train and evaluate models
python model.py

# Run backtest
python backtest.py
python intraday_backtest.py
```