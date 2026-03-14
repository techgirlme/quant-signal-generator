import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
TICKER = "RELIANCE.NS"
PERIOD = "2y"
RF_RATE = 0.065 / 252   # Daily risk-free rate

print(f"Fetching data for {TICKER}...")
df = yf.download(TICKER, period=PERIOD, auto_adjust=True)
df = df[["Close", "High", "Low", "Volume"]].copy()
df.columns = ["close", "high", "low", "volume"]
df = df.dropna()
print(f"  {len(df)} trading days loaded\n")

# ── Indicators ─────────────────────────────────────────────────────────────────

# 1. RSI


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 2. MACD


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# 3. Bollinger Bands


def compute_bollinger(series, window=20, n_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (series - lower) / (upper - lower)
    return upper, sma, lower, pct_b

# 4. Momentum


def compute_momentum(series, window=20):
    return series.pct_change(window)


df["rsi"] = compute_rsi(df["close"])
df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
df["bb_upper"], df["bb_mid"], df["bb_lower"], df["pct_b"] = compute_bollinger(
    df["close"])
df["momentum"] = compute_momentum(df["close"])
df = df.dropna()

# ── Signal Generation ──────────────────────────────────────────────────────────
# Each indicator votes: +1 = bullish, -1 = bearish, 0 = neutral

df["sig_rsi"] = np.where(df["rsi"] < 35, 1, np.where(df["rsi"] > 65, -1, 0))
df["sig_macd"] = np.where(df["macd"] > df["macd_signal"], 1, -1)
df["sig_bb"] = np.where(df["pct_b"] < 0.2, 1,
                        np.where(df["pct_b"] > 0.8, -1, 0))
df["sig_momentum"] = np.where(
    df["momentum"] > 0.02, 1, np.where(df["momentum"] < -0.02, -1, 0))

# Composite signal score (-4 to +4)
df["composite"] = df["sig_rsi"] + df["sig_macd"] + \
    df["sig_bb"] + df["sig_momentum"]

# Trade signal: buy if score >= 2, sell if score <= -2
df["signal"] = np.where(df["composite"] >= 2, 1,
                        np.where(df["composite"] <= -2, -1, 0))
df["position"] = df["signal"].shift(1).fillna(0)

# ── Backtest ───────────────────────────────────────────────────────────────────
df["daily_return"] = df["close"].pct_change()
df["strategy_return"] = df["position"] * df["daily_return"]
df["cum_market"] = (1 + df["daily_return"]).cumprod()
df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()

# Performance metrics
total_return = df["cum_strategy"].iloc[-1] - 1
market_return = df["cum_market"].iloc[-1] - 1
ann_return = (1 + total_return) ** (252 / len(df)) - 1
ann_vol = df["strategy_return"].std() * np.sqrt(252)
sharpe = (ann_return - RF_RATE * 252) / ann_vol
max_dd = ((df["cum_strategy"] - df["cum_strategy"].cummax()) /
          df["cum_strategy"].cummax()).min()
trades = df["signal"].diff().abs().sum() / 2
win_rate = (df["strategy_return"][df["strategy_return"] != 0] > 0).mean()

print("=" * 50)
print("  STRATEGY PERFORMANCE REPORT")
print("=" * 50)
print(f"  Total Strategy Return : {total_return:.2%}")
print(f"  Buy & Hold Return     : {market_return:.2%}")
print(f"  Annualised Return     : {ann_return:.2%}")
print(f"  Annualised Volatility : {ann_vol:.2%}")
print(f"  Sharpe Ratio          : {sharpe:.3f}")
print(f"  Maximum Drawdown      : {max_dd:.2%}")
print(f"  Number of Trades      : {int(trades)}")
print(f"  Win Rate              : {win_rate:.2%}")
print("=" * 50)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.4)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

# Panel 1: Price + Bollinger Bands + signals
ax1.plot(df.index, df["close"], color="#1F2D3D", linewidth=1, label="Price")
ax1.plot(df.index, df["bb_upper"], color="#B4B2A9",
         linewidth=0.8, linestyle="--", label="BB Upper")
ax1.plot(df.index, df["bb_mid"],   color="#888780",
         linewidth=0.8, linestyle="--", label="BB Mid")
ax1.plot(df.index, df["bb_lower"], color="#B4B2A9",
         linewidth=0.8, linestyle="--", label="BB Lower")
ax1.fill_between(df.index, df["bb_upper"],
                 df["bb_lower"], alpha=0.1, color="#7F77DD")

buy_signals = df[df["signal"] == 1]
sell_signals = df[df["signal"] == -1]
ax1.scatter(buy_signals.index,  buy_signals["close"],  marker="^",
            color="#1D9E75", s=60, zorder=5, label="Buy Signal")
ax1.scatter(sell_signals.index, sell_signals["close"], marker="v",
            color="#D85A30", s=60, zorder=5, label="Sell Signal")
ax1.set_title(
    f"{TICKER.replace('.NS', '')} — Price with Bollinger Bands & Signals", fontweight="bold")
ax1.set_ylabel("Price (INR)")
ax1.legend(loc="upper left", fontsize=8, ncol=3)

# Panel 2: RSI
ax2.plot(df.index, df["rsi"], color="#7F77DD", linewidth=1)
ax2.axhline(70, color="#D85A30", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.axhline(30, color="#1D9E75", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.fill_between(df.index, df["rsi"], 70, where=(
    df["rsi"] >= 70), alpha=0.3, color="#D85A30")
ax2.fill_between(df.index, df["rsi"], 30, where=(
    df["rsi"] <= 30), alpha=0.3, color="#1D9E75")
ax2.set_title("RSI (14)", fontweight="bold")
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)

# Panel 3: MACD
ax3.plot(df.index, df["macd"],        color="#1D9E75",
         linewidth=1,   label="MACD")
ax3.plot(df.index, df["macd_signal"], color="#D85A30",
         linewidth=1,   label="Signal")
ax3.bar(df.index,  df["macd_hist"],   color=np.where(
    df["macd_hist"] >= 0, "#1D9E75", "#D85A30"), alpha=0.5, label="Histogram")
ax3.axhline(0, color="#888780", linewidth=0.5)
ax3.set_title("MACD (12, 26, 9)", fontweight="bold")
ax3.set_ylabel("MACD")
ax3.legend(loc="upper left", fontsize=8)

# Panel 4: Cumulative returns
ax4.plot(df.index, (df["cum_strategy"] - 1) * 100, color="#1D9E75",
         linewidth=1.5, label=f"Strategy ({total_return:.1%})")
ax4.plot(df.index, (df["cum_market"] - 1) * 100, color="#7F77DD",
         linewidth=1.5, label=f"Buy & Hold ({market_return:.1%})")
ax4.fill_between(df.index, (df["cum_strategy"]-1)*100, (df["cum_market"]-1)*100,
                 where=(df["cum_strategy"] >= df["cum_market"]),
                 alpha=0.2, color="#1D9E75", label="Strategy outperforms")
ax4.fill_between(df.index, (df["cum_strategy"]-1)*100, (df["cum_market"]-1)*100,
                 where=(df["cum_strategy"] < df["cum_market"]),
                 alpha=0.2, color="#D85A30", label="Strategy underperforms")
ax4.axhline(0, color="#888780", linewidth=0.5)
ax4.set_title("Cumulative Returns — Strategy vs Buy & Hold", fontweight="bold")
ax4.set_ylabel("Return (%)")
ax4.legend(loc="upper left", fontsize=8)

plt.suptitle(f"Quantitative Signal Generator — {TICKER.replace('.NS', '')} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}",
             fontsize=13, fontweight="bold")

save_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "signal_output.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nChart saved as signal_output.png")
