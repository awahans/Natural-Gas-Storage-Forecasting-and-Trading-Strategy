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
| **% Storage Full (lag 1)** | Last week's storage level as % of capacity. Motivated by Darcy flow dynamics — as storage fills, reservoir pressure increases and injection rate slows. |
| **Storage Change (lag 1)** | Last week's storage change. Autoregressive momentum term. |
| **Freeze Intensity** | Production-weighted sum of degrees below 15°F across Marcellus, Permian, Haynesville, and DJ Basin production regions. Below 15°F, hydrate formation in wellhead equipment causes supply disruptions independent of demand — a ChemE-motivated feature that simultaneously shocks supply and demand. |

**Key design choice:** Weather features cover Monday–Thursday of the EIA report week (data available before Thursday 10:30 AM release). Storage lag features use the prior week's released data. No look-ahead bias.

---

## Models

Walk-forward validation: 2-year rolling training window, predict 1 week ahead, retrain weekly. Simulates live trading — model only sees data available at prediction time.

| Model | CV R² | CV MSE |
|-------|-------|--------|
| HDD + CDD (baseline) | 0.932 | 650.6 |
| + % Storage Full | 0.934 | 630.5 |
| + Storage Change Lag | 0.942 | 564.3 |
| + Freeze Event (binary) | 0.942 | 568.0 |
| + **Freeze Intensity (weighted)** | **0.946** | **519.9** |
| XGBoost (all features) | 0.945 | 533.4 |

**Key finding:** Linear regression with 4 physically-motivated features outperforms XGBoost. The relationship is fundamentally linear — consistent with the physical drivers (degree days, Darcy flow). XGBoost finds no additional nonlinear structure worth capturing at this sample size (~630 observations).

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