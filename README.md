# MarketPULSE

**Real-Time Stock Insights, Trends & Alerts**

A web application that predicts the *direction* of a stock's next-day close using a stacked LSTM trained on price history and technical indicators, and presents it through a dashboard with watchlists, charts, a live news feed, and alerts.

> **This is a decision-support and educational tool, not investment advice.** Measured next-day directional accuracy is **52.93%** — marginally above a coin flip, which is what honest short-horizon price prediction looks like. Do not trade real money on its output. See [Limitations](#limitations).

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The model](#the-model)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [References](#references)
- [License & disclaimer](#license--disclaimer)

---

## What it does

| Feature | Description |
|---|---|
| **Next-day direction prediction** | A stacked LSTM outputs the probability that a ticker closes higher on the next trading session. |
| **Trading signal** | The probability is thresholded into a Buy / Sell / Hold suggestion. |
| **Technical indicators** | RSI, MACD, Stochastic Oscillator, daily returns, and volume are computed on the fly and fed to the model as features. |
| **Interactive dashboard** | Watchlist of favourite tickers showing current vs. predicted price, plus a rising/falling breakdown. |
| **Live news feed** | Financial headlines refreshed automatically, with a cached fallback when the upstream feed is unavailable. |
| **Alerts** | Notifications for price breakouts, trend reversals, and high-volatility conditions on watchlist tickers. |
| **Accounts** | Registration and login, with per-user favourites persisted in MySQL. |
| **Backtest view** | Cumulative-return curve comparing the LSTM directional strategy against passive buy-and-hold over the test window. |

---

## Architecture

Three layers, loosely coupled so the model can be retrained and swapped without touching the UI.

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION  ·  HTML5 / CSS3 / JavaScript                 │
│  landing · auth · dashboard · charts · news · alerts        │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (JSON)
┌───────────────────────────▼─────────────────────────────────┐
│  APPLICATION  ·  Flask (Python)                             │
│                                                             │
│   ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│   │  Ingest    │→ │ Preprocess & │→ │  Inference        │   │
│   │  yfinance  │  │ feature eng. │  │  Keras LSTM       │   │
│   └────────────┘  │ RSI/MACD/    │  │  → P(up)          │   │
│                   │ Stoch, scale,│  └─────────┬─────────┘   │
│                   │ 60-day seq.  │            │             │
│                   └──────────────┘            ▼             │
│                   ┌──────────────┐  ┌───────────────────┐   │
│                   │ Alert rule   │← │ Signal mapper     │   │
│                   │ engine       │  │ Buy / Sell / Hold │   │
│                   └──────────────┘  └───────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  DATA  ·  MySQL (users, favourites, prediction history)     │
│           Yahoo Finance API · news API                      │
└─────────────────────────────────────────────────────────────┘
```

### Request flow

1. User submits a ticker.
2. Backend pulls OHLCV history via `yfinance`.
3. Missing values handled; RSI, MACD, Stochastic Oscillator, daily return and volume computed with the `ta` library.
4. Features scaled with `MinMaxScaler` **fitted on the training split only**, then reshaped into rolling 60-day sequences.
5. The saved Keras model returns P(next close > today's close).
6. Probability is mapped to Buy / Sell / Hold and rendered with charts and indicator overlays.
7. Alert rules evaluate the result against the user's watchlist thresholds.

---

## The model

Binary classifier over 60-day windows. The target is the sign of the next day's close-to-close change.

```
Input  (60 timesteps × n features)
  ↓
LSTM(64, return_sequences=True)
  ↓  BatchNormalization  →  Dropout(0.2)
LSTM(32, return_sequences=False)
  ↓  BatchNormalization  →  Dropout(0.2)
Dense(16, activation='relu')
  ↓
Dense(1,  activation='sigmoid')     →  P(Up)
```

| Setting | Value |
|---|---|
| Sequence length | 60 trading days |
| Features | Close, Volume, Daily Return, RSI, MACD, Stochastic Oscillator |
| Scaler | `MinMaxScaler` (fit on train split only) |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Callbacks | `EarlyStopping`, `ReduceLROnPlateau` |
| Reference training data | TSLA, 2015-01-01 → 2026-07-01 |
| Split | Chronological — no shuffling, no random split |

**Why chronological splitting matters.** Random train/test splits on time-series data leak future information backwards and produce inflated accuracy. The 52.93% reported below is a leak-free number; treat any stock-prediction project claiming 95%+ directional accuracy as suspect.

---

## Tech stack

**Frontend** — HTML5, CSS3, vanilla JavaScript

**Backend & ML**

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Flask |
| Deep learning | TensorFlow / Keras |
| Data handling | Pandas, NumPy |
| Preprocessing & metrics | scikit-learn |
| Market data | Yahoo Finance (`yfinance`) |
| Technical indicators | `ta` |
| Plotting | Matplotlib |
| Database | MySQL |

**Development** — VS Code, Google Colab / Jupyter (model training)

**System requirements** — Windows 10/11, Linux or macOS · Intel Core i5 or better · 8 GB RAM (16 GB recommended) · ~10 GB free disk · internet connection for market data

---

## Getting started

### Prerequisites

- Python 3.10 or above
- MySQL Server 8.0+
- pip and (recommended) `venv`

### Installation

```bash
git clone https://github.com/<your-username>/marketpulse.git
cd marketpulse

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Minimum dependency set:

```
flask
tensorflow
pandas
numpy
scikit-learn
yfinance
ta
matplotlib
mysql-connector-python
python-dotenv
```

### Database setup

```sql
CREATE DATABASE marketpulse;
USE marketpulse;

CREATE TABLE users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    email         VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE favourites (
    id       INT AUTO_INCREMENT PRIMARY KEY,
    user_id  INT NOT NULL,
    ticker   VARCHAR(16) NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_user_ticker (user_id, ticker),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE predictions (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    ticker         VARCHAR(16) NOT NULL,
    predicted_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    probability    FLOAT NOT NULL,
    direction      ENUM('UP','DOWN') NOT NULL,
    recommendation VARCHAR(16),
    INDEX idx_ticker_time (ticker, predicted_at)
);
```

### Train the model

```bash
python train.py --ticker TSLA --start 2015-01-01 --end 2026-07-01
```

This writes the trained model and the fitted scaler to `models/`.

### Run

```bash
flask --app app run --debug
```

Open `http://127.0.0.1:5000`.

---

## Configuration

Secrets belong in a `.env` file that is **not** committed. Add `.env` to `.gitignore`.

```env
FLASK_SECRET_KEY=change-me
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=marketpulse
MYSQL_PASSWORD=
MYSQL_DATABASE=marketpulse

NEWS_API_KEY=
MODEL_PATH=models/lstm_model.keras
SCALER_PATH=models/scaler.pkl
SEQ_LENGTH=60
```

---

## Project structure

```
marketpulse/
├── app.py                  # Flask entry point and routes
├── train.py                # Model training pipeline
├── requirements.txt
├── .env.example
├── src/
│   ├── data.py             # yfinance ingestion
│   ├── features.py         # RSI, MACD, Stochastic, returns, scaling
│   ├── model.py            # LSTM definition, load/save
│   ├── predict.py          # Inference and Buy/Sell/Hold mapping
│   ├── alerts.py           # Alert rule engine
│   └── db.py               # MySQL access layer
├── models/                 # Saved model + scaler (gitignored)
├── static/
│   ├── css/
│   └── js/
├── templates/              # Jinja templates
└── notebooks/              # Training and evaluation notebooks
```

---

## Usage

1. **Register or log in.**
2. **Search a ticker** (e.g. `AAPL`, `TSLA`, `GOOGL`) on the landing page.
3. Read the **current price, predicted direction, and probability**. A probability near 0.50 means the model has no real opinion — treat it as such.
4. **Star the ticker** to add it to your dashboard watchlist.
5. Review **charts and indicator overlays** to see whether the technical picture agrees with the model.
6. Check the **news feed** for events the model cannot see.

---

## Results

Evaluated on a held-out chronological test window of 512 trading days (TSLA).

**Next-day directional accuracy: 52.93%**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Down / Flat | 0.51 | 0.55 | 0.53 | 245 |
| Up | 0.55 | 0.51 | 0.53 | 267 |
| **Macro avg** | **0.53** | **0.53** | **0.53** | **512** |

**Reading these numbers honestly.** The majority class (Up) is 52.1% of the test set, so a model that always predicted "Up" would score ≈52.1%. The LSTM's 52.93% is therefore roughly **0.8 percentage points above the naive baseline** — a small edge, within the range that could be noise on a single ticker and a single test window.

The cumulative-return backtest shows the directional strategy tracking or modestly exceeding buy-and-hold across the test period, but this is a **frictionless** backtest: it excludes brokerage fees, bid-ask spread, slippage, and taxes, any of which would likely erase an edge this thin.

That the number is barely above chance is not a bug in the implementation. It is the expected result for next-day directional forecasting from price history alone, and it is consistent with the weak-form Efficient Market Hypothesis [1].

---

## Limitations

Stated plainly, because knowing where a system breaks is part of the system.

- **The edge is not established.** 52.93% on one ticker, one test window, is not evidence of a durable signal. Establishing one would need multiple tickers, multiple time periods, walk-forward validation, and confidence intervals.
- **Frictionless backtest.** No transaction costs, spread, slippage, or market impact are modelled. Daily-turnover strategies are especially sensitive to these.
- **Single-ticker training.** The model is trained per-ticker; it does not generalise across instruments without retraining.
- **No regime-shift handling.** A model trained through mid-2026 will degrade as market regimes change. There is no automated drift detection or scheduled retraining.
- **Sentiment is not in the model.** The news feed is displayed to the user but is *not* an input feature. Sentiment integration is future work, not a current capability.
- **Data quality.** Yahoo Finance data has occasional gaps, split/dividend adjustment quirks, and no guarantee of point-in-time accuracy.
- **The model cannot see events.** Earnings surprises, macro announcements, and geopolitical shocks dominate short-horizon moves and are absent from the feature set.
- **Not hardened for production.** Rate limiting, CSRF protection, session hardening, and input validation on ticker symbols would all need attention before any public deployment.

---

## Testing

| Level | Coverage |
|---|---|
| Unit | Auth, data retrieval, preprocessing, indicator computation (RSI / MACD / Stochastic), prediction module, dashboard rendering |
| Integration | Frontend ↔ backend, backend ↔ Yahoo Finance, backend ↔ model, model ↔ dashboard |
| Functional | Registration, login, ticker search, prediction generation, recommendation output, chart rendering |
| Model | Accuracy, precision, recall, F1, confusion matrix, cumulative return |
| Performance | Data-fetch latency, inference latency, memory during model execution |
| UI | Navigation, responsive layout, graph rendering, input validation |

---

## Roadmap

- [ ] **News sentiment as a feature** — NLP over headlines, fed to the model rather than only displayed
- [ ] **Alternative architectures** — GRU, Bi-LSTM, attention-based LSTM, Temporal Convolutional Networks, Transformers
- [ ] **Walk-forward validation** across multiple tickers, with confidence intervals
- [ ] **Cost-aware backtesting** including fees and slippage
- [ ] **Streaming data** for continuous intraday prediction updates
- [ ] **Portfolio management** — holdings, P&L, position sizing
- [ ] **User-configurable alerts** with email / SMS / push delivery
- [ ] **Multi-market support** — international exchanges, crypto, commodities, ETFs
- [ ] **Scheduled retraining** with drift monitoring
- [ ] **Mobile apps** (Android / iOS)

---

## References

1. Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383–417.
2. Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). *Predicting Stock Market Index Using Fusion of Machine Learning Techniques.* Expert Systems with Applications, 42(4), 2162–2172.
3. Bhandari, H. N., Rimal, B., Pokharel, S., & Ghimire, A. (2022). *Predicting Stock Market Index Using LSTM Models.* Applied Soft Computing.
4. Zhang, J., Wang, X., Li, Y., & Chen, W. (2024). *A Hybrid Wavelet–ARIMA–LSTM Approach for Stock Market Forecasting.* Applied Soft Computing.
5. Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter Mood Predicts the Stock Market.* Journal of Computational Science, 2(1), 1–8.
6. Zou, J., Han, Y., Li, X., Wang, Z., & Liu, H. (2022). *Stock Market Prediction via Deep Learning Techniques: A Comprehensive Survey.* IEEE Access.
7. Nabipour, M., Nayyeri, P., Jabani, H., Shahab, S., & Mosavi, A. (2020). *Deep Learning Applications for Stock Market Prediction.* IEEE Access, 8, 159438–159447.

---

## License & disclaimer

Released under the MIT License. See `LICENSE`.

**Financial disclaimer.** MarketPULSE is a research and learning project. Its predictions are statistical estimates with accuracy barely above chance, and it does not constitute financial, investment, or trading advice. No liability is accepted for any loss arising from use of this software. Consult a qualified financial advisor before making investment decisions.
