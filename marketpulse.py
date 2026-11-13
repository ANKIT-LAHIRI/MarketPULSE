"""
MarketPULSE: next-day stock direction forecasting
=================================================
Predicts whether a stock closes higher tomorrow than today, using a stacked LSTM
and three honest baselines (majority class, Random Forest, XGBoost), all scored
on exactly the same chronological test days.

Why this is leakage-free
  * Chronological 70 / 10 / 20 split. Rows are never shuffled.
  * The scaler is fitted on training rows only.
  * A window for day t uses only rows t-59 .. t (past data) and predicts day t+1.
  * Early stopping uses the validation block, never the test block.

Run in Google Colab (paste into one cell) or locally:
    python marketpulse.py                       # MSFT, 3 LSTM seeds
    python marketpulse.py --ticker TSLA --seeds 5
    python marketpulse.py --no-lstm             # baselines only (fast)
"""

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


def ensure(package, module=None):
    """Install a package on first use (handy in Colab)."""
    try:
        __import__(module or package)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


for _pkg, _mod in [("yfinance", "yfinance"), ("ta", "ta"), ("xgboost", "xgboost"), ("scipy", "scipy")]:
    ensure(_pkg, _mod)

from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from xgboost import XGBClassifier


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class Config:
    ticker: str = "MSFT"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    seq_length: int = 60           # look-back window in trading days
    train_frac: float = 0.70
    val_frac: float = 0.10         # remaining 20% is the test block
    seeds: int = 3                 # LSTM runs with different seeds (mean and std reported)
    epochs: int = 60
    batch_size: int = 32
    seed: int = 42
    use_lstm: bool = True
    mock: bool = False             # synthetic prices, for offline testing only
    out: str = "marketpulse_results.json"


FEATURES = ["Return", "Volume_chg", "HL_range", "RSI", "MACD_diff", "Stoch"]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_prices(cfg: Config) -> pd.DataFrame:
    if cfg.mock:
        rng = np.random.default_rng(0)
        n = 2500
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
             "Volume": rng.integers(1_000_000, 9_000_000, n)},
            index=pd.bdate_range("2015-01-01", periods=n))
    import yfinance as yf
    df = yf.download(cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):          # newer yfinance returns two header rows
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"No price data downloaded for {cfg.ticker}.")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def build_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Six stationary features plus the next-day target."""
    close = prices["Close"]
    t = pd.DataFrame(index=prices.index)
    t["Close"] = close
    t["Return"] = close.pct_change()
    t["Volume_chg"] = np.log1p(prices["Volume"]).diff()
    t["HL_range"] = (prices["High"] - prices["Low"]) / close
    t["RSI"] = RSIIndicator(close, window=14).rsi()
    t["MACD_diff"] = MACD(close).macd_diff() / close           # scale-free
    t["Stoch"] = StochasticOscillator(prices["High"], prices["Low"], close).stoch()
    t["Next_return"] = close.pct_change().shift(-1)
    t["Target"] = (t["Next_return"] > 0).astype(float)
    t.loc[t["Next_return"].isna(), "Target"] = np.nan          # last day has no tomorrow
    t = t.replace([np.inf, -np.inf], np.nan)
    return t


def make_windows(x: np.ndarray, label_rows: np.ndarray, seq: int) -> np.ndarray:
    """One window per label row: rows [t-seq+1 .. t]. Uses past data only."""
    return np.stack([x[t - seq + 1: t + 1] for t in label_rows])


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def build_lstm(seq: int, n_features: int):
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    model = Sequential([
        Input(shape=(seq, n_features)),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(cfg, seed, Xtr, ytr, Xva, yva):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    set_seeds(seed)
    model = build_lstm(Xtr.shape[1], Xtr.shape[2])
    model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=cfg.epochs, batch_size=cfg.batch_size,
        shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                   ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)])
    return model


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(cfg: Config):
    table = build_table(load_prices(cfg))
    live_row = len(table) - 1                                   # most recent day (no label yet)
    clean = table.dropna(subset=FEATURES)
    clean_idx = clean.index
    labelled = clean.dropna(subset=["Target"])

    X_all = clean[FEATURES].values
    y_all = clean["Target"].fillna(-1).values.astype(int)       # -1 marks the unlabelled live day
    n_labelled = len(labelled)
    seq = cfg.seq_length

    train_end = int(n_labelled * cfg.train_frac)
    val_end = int(n_labelled * (cfg.train_frac + cfg.val_frac))

    scaler = MinMaxScaler().fit(X_all[:train_end])              # training rows only
    Xs = scaler.transform(X_all)

    rows_train = np.arange(seq - 1, train_end)
    rows_val = np.arange(train_end, val_end)
    rows_test = np.arange(val_end, n_labelled)
    y_train, y_val, y_test = y_all[rows_train], y_all[rows_val], y_all[rows_test]

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"{cfg.ticker}: {clean_idx[0].date()} to {clean_idx[n_labelled - 1].date()}  |  "
          f"train {len(rows_train)}  val {len(rows_val)}  test {len(rows_test)} days  |  "
          f"majority baseline {baseline:.1%}")

    results = {"config": asdict(cfg), "features": FEATURES, "n_test": int(len(rows_test)),
               "majority_baseline": float(baseline), "models": {}}

    # ---- baselines (tabular: today's six features -> tomorrow's direction)
    tab = {"train": (Xs[rows_train], y_train), "test": (Xs[rows_test], y_test)}
    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest (bagging)": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=cfg.seed, n_jobs=-1),
        "XGBoost (boosting)": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, random_state=cfg.seed, eval_metric="logloss"),
    }
    for name, model in baselines.items():
        model.fit(*tab["train"])
        results["models"][name] = score(y_test, model.predict(tab["test"][0]))

    # ---- LSTM (several seeds, because one run of a noisy model proves little)
    lstm_pred = None
    if cfg.use_lstm:
        Xtr = make_windows(Xs, rows_train, seq)
        Xva = make_windows(Xs, rows_val, seq)
        Xte = make_windows(Xs, rows_test, seq)
        accs, f1s, probs = [], [], []
        for k in range(cfg.seeds):
            model = train_lstm(cfg, cfg.seed + k, Xtr, y_train, Xva, y_val)
            p = model.predict(Xte, verbose=0).flatten()
            probs.append(p)
            pred = (p > 0.5).astype(int)
            accs.append(accuracy_score(y_test, pred))
            f1s.append(f1_score(y_test, pred, average="macro", zero_division=0))
            print(f"  LSTM seed {cfg.seed + k}: accuracy {accs[-1]:.2%}")
        lstm_pred = (np.mean(probs, axis=0) > 0.5).astype(int)
        results["models"]["Stacked LSTM (mean of seeds)"] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "seeds": cfg.seeds,
            "ensemble": score(y_test, lstm_pred)}

        # live forecast for the most recent trading day
        live_window = Xs[-seq:].reshape(1, seq, len(FEATURES))
        results["live_probability_up"] = float(model.predict(live_window, verbose=0)[0, 0])

    # ---- report
    k_correct = lambda acc: int(round(acc * len(y_test)))
    print("\n========== RESULTS (copy everything below to Claude) ==========")
    print(f"Ticker {cfg.ticker} | features {FEATURES} | test {len(y_test)} days | "
          f"majority baseline {baseline:.1%}")
    for name, m in results["models"].items():
        if "accuracy" in m:
            acc, f1, cm = m["accuracy"], m["macro_f1"], m["confusion"]
            p = binomtest(k_correct(acc), len(y_test), baseline, alternative="greater").pvalue
            print(f"\n{name}: accuracy {acc:.2%} | macro-F1 {f1:.3f} | "
                  f"vs baseline {acc - baseline:+.1%} (p={p:.2f})")
            print(f"  confusion [[TN FP] [FN TP]]: {cm}")
        else:
            print(f"\n{name}: accuracy {m['accuracy_mean']:.2%} +/- {m['accuracy_std']:.2%} "
                  f"over {m['seeds']} seeds | macro-F1 {m['macro_f1_mean']:.3f}")
            e = m["ensemble"]
            print(f"  seed-averaged ensemble: accuracy {e['accuracy']:.2%} | "
                  f"macro-F1 {e['macro_f1']:.3f} | confusion {e['confusion']}")
    if "live_probability_up" in results:
        print(f"\nLatest-day model output: P(up tomorrow) = {results['live_probability_up']:.1%} "
              "(a model output, not trading advice)")

    # ---- simple strategy check (no transaction costs; for context only)
    if lstm_pred is not None:
        r = labelled["Next_return"].values[rows_test]
        bh = float(np.prod(1 + r) - 1)
        strat = float(np.prod(1 + np.where(lstm_pred == 1, r, 0)) - 1)
        results["strategy"] = {"buy_and_hold": bh, "lstm_long_or_cash": strat}
        print(f"\nTest-period return: buy and hold {bh:.1%} | LSTM long-or-cash {strat:.1%} "
              "(no costs or slippage)")

    with open(cfg.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {cfg.out}")
    return results


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="MSFT")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--mock", action="store_true")
    a, _ = p.parse_known_args()                                 # tolerates Colab's extra arguments
    return Config(ticker=a.ticker, start=a.start, end=a.end, seeds=a.seeds,
                  use_lstm=not a.no_lstm, mock=a.mock)


if __name__ == "__main__":
    main(parse_args())
