pip install pandas numpy scikit-learn ta yfinance ccxt matplotlib seaborn
pip install tensorflow keras
import ccxt
exchange = ccxt.binance()
btc_data = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
import ta
df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
pip install stable-baselines3
pip install backtrader
exchange.creaimport pandas as pd
import yfinance as yf

# Load BTC data
df = yf.download("BTC-USD", start="2023-01-01", interval="1h")

# Calculate moving averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# Generate signals
df['Signal'] = 0
df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1   # Buy
df.loc[df['SMA50'] < df['SMA200'], 'Signal'] = -1  # Sell

print(df[['Close','SMA50','SMA200','Signal']].tail())
te_market_buy_order('BTC/USDT', amount=0.01)
import ta

# Compute RSI
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

# Signals
df['Signal'] = 0
df.loc[df['RSI'] < 30, 'Signal'] = 1   # Buy
df.loc[df['RSI'] > 70, 'Signal'] = -1  # Sell
# Bollinger Bands
bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['BB_high'] = bb.bollinger_hband()
df['BB_low'] = bb.bollinger_lband()

# Signals
df['Signal'] = 0
df.loc[df['Close'] > df['BB_high'], 'Signal'] = 1   # Breakout Buy
df.loc[df['Close'] < df['BB_low'], 'Signal'] = -1   # Breakdown Sell
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Features
df['Return'] = df['Close'].pct_change()
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['SMA'] = df['Close'].rolling(20).mean()
df = df.dropna()

X = df[['Return','RSI','SMA']]
y = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 if next candle up

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
# pip install yfinance pandas numpy ta

import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------
# Data loading (1H BTC)
# ----------------------
df = yf.download("BTC-USD", interval="1h", start="2023-01-01")
df = df.rename(columns=str.lower)
df = df[['open','high','low','close','volume']].dropna()

# ----------------------
# CRT: 4H range levels
# ----------------------
df_4h = df.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
df_4h['crt_high'] = df_4h['high'].shift(1)
df_4h['crt_low']  = df_4h['low'].shift(1)
df = df.join(df_4h[['crt_high','crt_low']], how='left')
df[['crt_high','crt_low']] = df[['crt_high','crt_low']].ffill()

# ----------------------
# Swing points (structure)
# ----------------------
def swing_points(data, lookback=5, lookforward=5):
    highs = data['high']
    lows = data['low']
    swing_high = (highs.shift(lookback).rolling(lookback).max() < highs) & \
                 (highs.shift(-lookforward).rolling(lookforward).max() < highs)
    swing_low  = (lows.shift(lookback).rolling(lookback).min() > lows) & \
                 (lows.shift(-lookforward).rolling(lookforward).min() > lows)
    return swing_high.fillna(False), swing_low.fillna(False)

df['swing_high'], df['swing_low'] = swing_points(df, 3, 3)

# Track last swing levels for BOS/CHoCH
df['last_swing_high'] = np.nan
df['last_swing_low'] = np.nan
last_high = np.nan
last_low = np.nan
for i in range(len(df)):
    if df['swing_high'].iloc[i]:
        last_high = df['high'].iloc[i]
    if df['swing_low'].iloc[i]:
        last_low = df['low'].iloc[i]
    df.at[df.index[i], 'last_swing_high'] = last_high
    df.at[df.index[i], 'last_swing_low'] = last_low

# ----------------------
# Liquidity sweep (CRT)
# ----------------------
# Bullish sweep: low pierces crt_low, close back above crt_low
df['bull_sweep'] = (df['low'] < df['crt_low']) & (df['close'] > df['crt_low'])
# Bearish sweep: high pierces crt_high, close back below crt_high
df['bear_sweep'] = (df['high'] > df['crt_high']) & (df['close'] < df['crt_high'])

# ----------------------
# BOS/CHoCH confirmation
# ----------------------
df['bull_bos'] = df['close'] > df['last_swing_high']
df['bear_bos'] = df['close'] < df['last_swing_low']

# ----------------------
# Fair Value Gap (simple 3-candle)
# ----------------------
# Bullish FVG at i if low[i+1] > high[i-1] (gap in between preceding and following)
df['bull_fvg_top'] = np.nan
df['bull_fvg_bot'] = np.nan
df['bear_fvg_top'] = np.nan
df['bear_fvg_bot'] = np.nan

for i in range(1, len(df)-1):
    prev_high = df['high'].iloc[i-1]
    next_low  = df['low'].iloc[i+1]
    prev_low  = df['low'].iloc[i-1]
    next_high = df['high'].iloc[i+1]

    # Bullish gap
    if next_low > prev_high:
        df.at[df.index[i], 'bull_fvg_top'] = next_low
        df.at[df.index[i], 'bull_fvg_bot'] = prev_high

    # Bearish gap
    if next_high < prev_low:
        df.at[df.index[i], 'bear_fvg_top'] = prev_low
        df.at[df.index[i], 'bear_fvg_bot'] = next_high

# Carry forward latest FVG until mitigated
df[['bull_fvg_top','bull_fvg_bot']] = df[['bull_fvg_top','bull_fvg_bot']].ffill()
df[['bear_fvg_top','bear_fvg_bot']] = df[['bear_fvg_top','bear_fvg_bot']].ffill()

# Mitigation check: if price trades into the gap, consider it "available" for entries
def in_bull_fvg(row):
    return (row['bull_fvg_bot'] is not np.nan) and (row['low'] <= row['bull_fvg_top']) and (row['low'] >= row['bull_fvg_bot'])

def in_bear_fvg(row):
    return (row['bear_fvg_bot'] is not np.nan) and (row['high'] >= row['bear_fvg_bot']) and (row['high'] <= row['bear_fvg_top'])

# ----------------------
# Entry logic: CRT sweep + BOS + FVG retest
# ----------------------
risk_perc = 0.01  # 1% risk per trade
initial_equity = 100_000
equity = initial_equity
position = 0  # +qty for long, -qty for short
entry_price = None
stop_price = None
target_price = None
trade_log = []

for i in range(2, len(df)):
    row = df.iloc[i]
    ts = df.index[i]

    # Flatten if target/stop hit
    if position != 0:
        if position > 0:
            # Long: stop or target
            if row['low'] <= stop_price:
                pnl = (stop_price - entry_price) * position
                equity += pnl
                trade_log.append({'time': ts, 'side': 'long', 'exit': 'stop', 'pnl': pnl, 'equity': equity})
                position = 0
            elif row['high'] >= target_price:
                pnl = (target_price - entry_price) * position
                equity += pnl
                trade_log.append({'time': ts, 'side': 'long', 'exit': 'target', 'pnl': pnl, 'equity': equity})
                position = 0
        else:
            # Short
            if row['high'] >= stop_price:
                pnl = (entry_price - stop_price) * (-position)
                equity += pnl
                trade_log.append({'time': ts, 'side': 'short', 'exit': 'stop', 'pnl': pnl, 'equity': equity})
                position = 0
            elif row['low'] <= target_price:
                pnl = (entry_price - target_price) * (-position)
                equity += pnl
                trade_log.append({'time': ts, 'side': 'short', 'exit': 'target', 'pnl': pnl, 'equity': equity})
                position = 0

    # If flat, look for setups
    if position == 0:
        # Bullish setup: sweep + BOS + retest into bullish FVG
        bull_setup = row['bull_sweep'] and row['bull_bos']
        bear_setup = row['bear_sweep'] and row['bear_bos']

        # Entry signals: retest into FVG area
        # Basic retest signal: current low trades into the FVG bounds
        bull_retest = False
        bear_retest = False
        if not np.isnan(row['bull_fvg_top']) and not np.isnan(row['bull_fvg_bot']):
            bull_retest = (row['low'] <= row['bull_fvg_top']) and (row['low'] >= row['bull_fvg_bot'])
        if not np.isnan(row['bear_fvg_top']) and not np.isnan(row['bear_fvg_bot']):
            bear_retest = (row['high'] >= row['bear_fvg_bot']) and (row['high'] <= row['bear_fvg_top'])

        if bull_setup and bull_retest:
            entry_price = row['close']
            # Stop below sweep low or FVG bottom
            stop_price = min(row['low'], row['bull_fvg_bot']) if not np.isnan(row['bull_fvg_bot']) else row['low']
            # Target at CRT high
            target_price = row['crt_high']
            # Position size by risk
            risk_per_unit = entry_price - stop_price
            if risk_per_unit > 0:
                risk_amount = equity * risk_perc
                qty = risk_amount / risk_per_unit
                position = qty
                trade_log.append({'time': ts, 'side': 'long', 'entry': entry_price, 'stop': stop_price, 'target': target_price, 'qty': qty, 'equity': equity})

        elif bear_setup and bear_retest:
            entry_price = row['close']
            stop_price = max(row['high'], row['bear_fvg_top']) if not np.isnan(row['bear_fvg_top']) else row['high']
            target_price = row['crt_low']
            risk_per_unit = stop_price - entry_price
            if risk_per_unit > 0:
                risk_amount = equity * risk_perc
                qty = risk_amount / risk_per_unit
                position = -qty
                trade_log.append({'time': ts, 'side': 'short', 'entry': entry_price, 'stop': stop_price, 'target': target_price, 'qty': qty, 'equity': equity})

# ----------------------
# Metrics
# ----------------------
trades = pd.DataFrame(trade_log)
wins = trades[trades['exit'] == 'target'].shape[0] if 'exit' in trades.columns else 0
losses = trades[trades['exit'] == 'stop'].shape[0] if 'exit' in trades.columns else 0
win_rate = wins / max(1, (wins + losses))
final_equity = equity

print("Trades:", len(trades))
print("Wins:", wins, "Losses:", losses, "Win rate:", round(win_rate*100, 2), "%")
print("Final equity:", round(final_equity, 2))
import yfinance as yf
import pandas as pd
from smartmoneyconcepts import smc

# 1. Load BTC data
df = yf.download("BTC-USD", interval="1h", start="2023-01-01")
df = df.rename(columns=str.lower)  # smc expects lowercase: open, high, low, close, volume

# 2. Detect swing highs/lows
swings = smc.swing_highs_lows(df, swing_length=20)

# 3. Break of Structure (BOS) & Change of Character (CHoCH)
bos = smc.bos_choch(df, swings, close_break=True)

# 4. Fair Value Gaps (FVG)
fvg = smc.fvg(df, join_consecutive=True)

# 5. Order Blocks (OB)
ob = smc.ob(df, swings, close_mitigation=False)

# Merge into one DataFrame
signals = pd.concat([df, swings, bos, fvg, ob], axis=1)

print(signals.tail(20))
//@version=5
strategy("MA Crossover with Alerts", overlay=true, margin_long=100, margin_short=100)

// Inputs
shortLen = input.int(50, "Short MA Length")
longLen  = input.int(200, "Long MA Length")

// Moving Averages
shortMA = ta.sma(close, shortLen)
longMA  = ta.sma(close, longLen)

// Plot
plot(shortMA, color=color.blue)
plot(longMA, color=color.red)

// Entry conditions
longCond  = ta.crossover(shortMA, longMA)
shortCond = ta.crossunder(shortMA, longMA)

// Strategy entries
if (longCond)
    strategy.entry("Long", strategy.long)
    alert("LONG_SIGNAL", alert.freq_once_per_bar_close)

if (shortCond)
    strategy.entry("Short", strategy.short)
    alert("SHORT_SIGNAL", alert.freq_once_per_bar_close)
    {
  "signal": "LONG_SIGNAL",
  "symbol": "{{ticker}}",
  "price": "{{close}}"
}
from flask import Flask, request
import ccxt

app = Flask(__name__)

# Connect to exchange
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    signal = data.get("signal")
    symbol = "BTC/USDT"

    if signal == "LONG_SIGNAL":
        exchange.create_market_buy_order(symbol, 0.01)
    elif signal == "SHORT_SIGNAL":
        exchange.create_market_sell_order(symbol, 0.01)

    return {"status": "ok"}

if __name__ == '__main__':
    app.run(port=5000)
    {
  "signal": "LONG_SIGNAL",
  "symbol": "{{ticker}}",
  "price": "{{close}}"
}
from flask import Flask, request
import ccxt

app = Flask(__name__)

# Connect to exchange
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    signal = data.get("signal")
    symbol = "BTC/USDT"

    if signal == "LONG_SIGNAL":
        exchange.create_market_buy_order(symbol, 0.01)
    elif signal == "SHORT_SIGNAL":
        exchange.create_market_sell_order(symbol, 0.01)

    return {"status": "ok"}

if __name__ == '__main__':
    app.run(port=5000)
    //@version=5
strategy("CRT + ICT/SMC Strategy", overlay=true, margin_long=100, margin_short=100)

// === Inputs ===
crtTF     = input.timeframe("240", "CRT Range Timeframe") // 4H CRT
swingLen  = input.int(5, "Swing Length")
rr        = input.float(2.0, "Risk/Reward Ratio")

// === CRT High/Low ===
crtHigh = request.security(syminfo.tickerid, crtTF, high[1])
crtLow  = request.security(syminfo.tickerid, crtTF, low[1])
plot(crtHigh, "CRT High", color=color.red)
plot(crtLow, "CRT Low", color=color.green)

// === Swing High/Low ===
swingHigh = ta.pivothigh(high, swingLen, swingLen)
swingLow  = ta.pivotlow(low, swingLen, swingLen)

// === BOS/CHoCH ===
bosBull = close > nz(swingHigh[1])
bosBear = close < nz(swingLow[1])

// === Liquidity Sweeps ===
bullSweep = low < crtLow and close > crtLow
bearSweep = high > crtHigh and close < crtHigh

// === Fair Value Gap (basic 3-bar) ===
bullFVG = low[1] > high[3]
bearFVG = high[1] < low[3]

// === Entry Conditions ===
longCond  = bullSweep and bosBull and bullFVG
shortCond = bearSweep and bosBear and bearFVG

// === Strategy Entries ===
if (longCond)
    strategy.entry("Long", strategy.long)
    // Stop below CRT low, target CRT high
    stop = crtLow
    target = close + (close - stop) * rr
    strategy.exit("Long Exit", "Long", stop=stop, limit=target)
    alert("LONG_SIGNAL", alert.freq_once_per_bar_close)

if (shortCond)
    strategy.entry("Short", strategy.short)
    // Stop above CRT high, target CRT low
    stop = crtHigh
    target = close - (stop - close) * rr
    strategy.exit("Short Exit", "Short", stop=stop, limit=target)
    alert("SHORT_SIGNAL", alert.freq_once_per_bar_close)
    //@version=5
indicator("CRT + ICT/SMC Visuals", overlay=true, max_labels_count=500, max_lines_count=500)

// === Inputs ===
crtTF     = input.timeframe("240", "CRT Range Timeframe") // 4H CRT
swingLen  = input.int(5, "Swing Length")
showFVG   = input.bool(true, "Show Fair Value Gaps")
showOB    = input.bool(true, "Show Order Blocks")
showBOS   = input.bool(true, "Show BOS/CHoCH")
rr        = input.float(2.0, "Risk/Reward Target")

// === CRT High/Low ===
crtHigh = request.security(syminfo.tickerid, crtTF, high[1])
crtLow  = request.security(syminfo.tickerid, crtTF, low[1])
plot(crtHigh, "CRT High", color=color.red, style=plot.style_linebr)
plot(crtLow, "CRT Low", color=color.green, style=plot.style_linebr)

// === Swing High/Low ===
swingHigh = ta.pivothigh(high, swingLen, swingLen)
swingLow  = ta.pivotlow(low, swingLen, swingLen)

// === BOS/CHoCH ===
bosBull = close > nz(swingHigh[1])
bosBear = close < nz(swingLow[1])

if showBOS
    if bosBull
        label.new(bar_index, high, "BOS↑", color=color.green, style=label.style_label_up, textcolor=color.white)
    if bosBear
        label.new(bar_index, low, "BOS↓", color=color.red, style=label.style_label_down, textcolor=color.white)

// === Liquidity Sweeps ===
bullSweep = low < crtLow and close > crtLow
bearSweep = high > crtHigh and close < crtHigh

if bullSweep
    label.new(bar_index, low, "Sweep Low", color=color.green, style=label.style_label_up, textcolor=color.white)
if bearSweep
    label.new(bar_index, high, "Sweep High", color=color.red, style=label.style_label_down, textcolor=color.white)

// === Fair Value Gaps (basic 3-bar) ===
bullFVG = low[1] > high[3]
bearFVG = high[1] < low[3]

if showFVG
    if bullFVG
        box.new(left=bar_index-3, right=bar_index, top=low[1], bottom=high[3], bgcolor=color.new(color.green, 85))
    if bearFVG
        box.new(left=bar_index-3, right=bar_index, top=low[3], bottom=high[1], bgcolor=color.new(color.red, 85))

// === Order Blocks (basic: last opposite candle before BOS) ===
var float obHigh = na
var float obLow  = na

if showOB
    if bosBull
        // Bullish OB = last down candle before BOS
        obHigh := high[1]
        obLow  := low[1]
        box.new(left=bar_index-1, right=bar_index+10, top=obHigh, bottom=obLow, bgcolor=color.new(color.green, 90))
    if bosBear
        // Bearish OB = last up candle before BOS
        obHigh := high[1]
        obLow  := low[1]
        box.new(left=bar_index-1, right=bar_index+10, top=obHigh, bottom=obLow, bgcolor=color.new(color.red, 90))

// === Entry Conditions ===
longCond  = bullSweep and bosBull and bullFVG
shortCond = bearSweep and bosBear and bearFVG

if longCond
    alert("LONG_SIGNAL", alert.freq_once_per_bar_close)
if shortCond
    alert("SHORT_SIGNAL", alert.freq_once_per_bar_close)
    # pip install flask ccxt

from flask import Flask, request
import ccxt

app = Flask(__name__)

# === Configure your exchange ===
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

symbol = "BTC/USDT"   # Trading pair
trade_size = 0.01     # Position size in BTC

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    signal = data.get("signal")

    if signal == "LONG_SIGNAL":
        order = exchange.create_market_buy_order(symbol, trade_size)
        print("Long order executed:", order)

    elif signal == "SHORT_SIGNAL":
        order = exchange.create_market_sell_order(symbol, trade_size)
        print("Short order executed:", order)

    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    ngrok http 5000
    alert("LONG_SIGNAL", alert.freq_once_per_bar_close)
alert("SHORT_SIGNAL", alert.freq_once_per_bar_close)
{ "signal": "LONG_SIGNAL" }
# pip install flask ccxt

from flask import Flask, request
import ccxt, time

app = Flask(__name__)

# === Exchange Setup ===
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

symbol = "BTC/USDT"
equity = 10000        # starting account equity (update dynamically if you want)
risk_per_trade = 0.01 # risk 1% of equity per trade
daily_loss_limit = 0.05 # stop trading if >5% equity lost in a day

# Track daily PnL
daily_start_equity = equity
daily_loss = 0

def get_price():
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

def calc_position_size(entry, stop, equity):
    risk_amount = equity * risk_per_trade
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 0
    size = risk_amount / stop_distance
    return round(size, 5)

@app.route('/webhook', methods=['POST'])
def webhook():
    global equity, daily_loss, daily_start_equity

    data = request.json
    signal = data.get("signal")
    price = get_price()

    # Check daily loss limit
    if (equity - daily_start_equity) / daily_start_equity <= -daily_loss_limit:
        print("Daily loss limit reached. No more trades today.")
        return {"status": "blocked"}

    if signal == "LONG_SIGNAL":
        stop = price * 0.99   # 1% stop below entry (example)
        target = price * 1.02 # 2% target (example)
        size = calc_position_size(price, stop, equity)
        if size > 0:
            order = exchange.create_market_buy_order(symbol, size)
            print("Long order executed:", order)
            # Place OCO (One-Cancels-Other) order for stop-loss + take-profit
            exchange.create_order(symbol, 'limit', 'sell', size, target,
                                  {'stopPrice': stop, 'reduceOnly': True})

    elif signal == "SHORT_SIGNAL":
        stop = price * 1.01   # 1% stop above entry
        target = price * 0.98 # 2% target
        size = calc_position_size(price, stop, equity)
        if size > 0:
            order = exchange.create_market_sell_order(symbol, size)
            print("Short order executed:", order)
            exchange.create_order(symbol, 'limit', 'buy', size, target,
                                  {'stopPrice': stop, 'reduceOnly': True})

    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
   # pip install flask ccxt

from flask import Flask, request
import ccxt, datetime

app = Flask(__name__)

# === Exchange Setup ===
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

# === Portfolio Settings ===
portfolio_equity = 10000          # starting equity (fetch dynamically if desired)
risk_per_trade = 0.01             # 1% risk per trade
daily_loss_limit = 0.05           # stop trading if >5% equity lost in a day
max_open_positions = 3            # cap simultaneous trades

# Track daily PnL
daily_start_equity = portfolio_equity
open_positions = {}

def get_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

def calc_position_size(entry, stop, equity):
    risk_amount = equity * risk_per_trade
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 0
    size = risk_amount / stop_distance
    return round(size, 5)

@app.route('/webhook', methods=['POST'])
def webhook():
    global portfolio_equity, daily_start_equity, open_positions

    data = request.json
    signal = data.get("signal")
    symbol = data.get("symbol", "BTC/USDT")  # TradingView can send symbol
    price = get_price(symbol)

    # Check daily loss limit
    if (portfolio_equity - daily_start_equity) / daily_start_equity <= -daily_loss_limit:
        print("Daily loss limit reached. No more trades today.")
        return {"status": "blocked"}

    # Check max open positions
    if len(open_positions) >= max_open_positions:
        print("Max open positions reached. Skipping trade.")
        return {"status": "blocked"}

    # Example stop/target logic (replace with ICT/SMC rules)
    if signal == "LONG_SIGNAL":
        stop = price * 0.99
        target = price * 1.02
        size = calc_position_size(price, stop, portfolio_equity)
        if size > 0:
            order = exchange.create_market_buy_order(symbol, size)
            open_positions[symbol] = {"side": "long", "entry": price, "stop": stop, "target": target, "size": size}
            print(f"Opened LONG {symbol} @ {price}, stop {stop}, target {target}")

    elif signal == "SHORT_SIGNAL":
        stop = price * 1.01
        target = price * 0.98
        size = calc_position_size(price, stop, portfolio_equity)
        if size > 0:
            order = exchange.create_market_sell_order(symbol, size)
            open_positions[symbol] = {"side": "short", "entry": price, "stop": stop, "target": target, "size": size}
            print(f"Opened SHORT {symbol} @ {price}, stop {stop}, target {target}")

    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000) 
    { "signal": "LONG_SIGNAL", "symbol": "{{ticker}}" }
    pip install streamlit plotly pandas ccxt
    import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import ccxt
import time

# === Exchange Setup ===
exchange = ccxt.binance({
    "apiKey": "YOUR_API_KEY",
    "secret": "YOUR_SECRET"
})

symbol_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📊 AI Trading Portfolio Dashboard")

# Sidebar controls
refresh_rate = st.sidebar.slider("Refresh (seconds)", 5, 60, 15)

# === Fetch account balance ===
def get_equity():
    balance = exchange.fetch_balance()
    return balance['total']['USDT']

# === Fetch open positions (simplified for spot) ===
def get_positions():
    positions = []
    for sym in symbol_list:
        ticker = exchange.fetch_ticker(sym)
        positions.append({
            "symbol": sym,
            "price": ticker['last'],
            "24h %": ticker['percentage']
        })
    return pd.DataFrame(positions)

# === Equity Curve (mocked with session state) ===
if "equity_curve" not in st.session_state:
    st.session_state.equity_curve = []

equity = get_equity()
st.session_state.equity_curve.append({"time": time.strftime("%H:%M:%S"), "equity": equity})

# === Layout ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Portfolio Equity")
    st.metric("Current Equity (USDT)", f"{equity:,.2f}")

    df_equity = pd.DataFrame(st.session_state.equity_curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_equity['time'], y=df_equity['equity'], mode='lines+markers'))
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Equity (USDT)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📌 Open Positions")
    pos_df = get_positions()
    st.dataframe(pos_df)

st.sidebar.info("Dashboard auto-refreshes every few seconds.")
st.experimental_rerun()
streamlit run dashboard.py
import csv, datetime

def log_trade(symbol, side, entry, stop, target, exit_price, pnl, equity):
    with open("trade_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.
        now(), symbol, side, entry, stop, target, exit_price, pnl, equity])
       import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")

st.title("📊 AI Trading Performance Dashboard")

# Load trade history
df = pd.read_csv("trade_log.csv", names=["time","symbol","side","entry","stop","target","exit","pnl","equity"])
df['time'] = pd.to_datetime(df['time'])

# === Key Metrics ===
total_trades = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]
win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
avg_rr = (wins['pnl'].mean() / abs(losses['pnl'].mean())) if len(losses) > 0 else None
max_drawdown = (df['equity'].cummax() - df['equity']).max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trades", total_trades)
col2.metric("Win Rate", f"{win_rate:.2f}%")
col3.metric("Avg R:R", f"{avg_rr:.2f}" if avg_rr else "N/A")
col4.metric("Max Drawdown", f"{max_drawdown:.2f} USDT")

# === Equity Curve ===
fig_equity = px.line(df, x="time", y="equity", title="Equity Curve")
st.plotly_chart(fig_equity, use_container_width=True)

# === PnL Distribution ===
fig_pnl = px.histogram(df, x="pnl", nbins=30, title="PnL Distribution")
st.plotly_chart(fig_pnl, use_container_width=True)

# === Trade History Table ===
st.subheader("📜 Trade History")
st.dataframe(df.tail(20)) 
streamlit run dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")
st.title("📊 AI Trading Performance Dashboard")

# Load trade history
df = pd.read_csv("trade_log.csv", names=["time","symbol","side","entry","stop","target","exit","pnl","equity"])
df['time'] = pd.to_datetime(df['time'])

# === Performance Metrics ===
total_trades = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]

win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
avg_rr = (wins['pnl'].mean() / abs(losses['pnl'].mean())) if len(losses) > 0 else None
max_drawdown = (df['equity'].cummax() - df['equity']).max()

# Sharpe & Sortino
returns = df['pnl'] / df['entry']  # simple returns per trade
mean_ret = returns.mean()
std_ret = returns.std()
downside_std = returns[returns < 0].std()

sharpe = (mean_ret / std_ret) * np.sqrt(len(df)) if std_ret > 0 else None
sortino = (mean_ret / downside_std) * np.sqrt(len(df)) if downside_std > 0 else None

# Profit Factor
gross_profit = wins['pnl'].sum()
gross_loss = abs(losses['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

# === Display Metrics ===
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", total_trades)
col2.metric("Win Rate", f"{win_rate:.2f}%")
col3.metric("Avg R:R", f"{avg_rr:.2f}" if avg_rr else "N/A")
col4.metric("Sharpe", f"{sharpe:.2f}" if sharpe else "N/A")
col5.metric("Sortino", f"{sortino:.2f}" if sortino else "N/A")
col6.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor else "N/A")

st.metric("Max Drawdown", f"{max_drawdown:.2f} USDT")

# === Equity Curve ===
fig_equity = px.line(df, x="time", y="equity", title="Equity Curve")
st.plotly_chart(fig_equity, use_container_width=True)

# === PnL Distribution ===
fig_pnl = px.histogram(df, x="pnl", nbins=30, title="PnL Distribution")
st.plotly_chart(fig_pnl, use_container_width=True)

# === Per-Symbol Breakdown ===
st.subheader("📌 Per-Symbol Performance")
symbol_perf = df.groupby("symbol").agg(
    trades=("pnl","count"),
    win_rate=("pnl", lambda x: (x>0).mean()*100),
    avg_pnl=("pnl","mean"),
    total_pnl=("pnl","sum")
).reset_index()

st.dataframe(symbol_perf)

fig_symbol = px.bar(symbol_perf, x="symbol", y="total_pnl", color="win_rate",
                    title="PnL by Symbol", text="win_rate")
st.plotly_chart(fig_symbol, use_container_width=True)

# === Trade History Table ===
st.subheader("📜 Trade History")
st.dataframe(df.tail(20))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Trading Performance Dashboard", layout="wide")
st.title("📊 AI Trading Performance Dashboard")

# Load trade history
df = pd.read_csv("trade_log.csv", names=["time","symbol","side","entry","stop","target","exit","pnl","equity"])
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

# === Session Mapping ===
def get_session(hour):
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London (pre‑NY overlap)"
    elif 13 <= hour < 21:
        return "New York"
    else:
        return "London/Asia Overlap"

df['session'] = df['hour'].apply(get_session)

# === Performance Metrics ===
total_trades = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] <= 0]

win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
avg_rr = (wins['pnl'].mean() / abs(losses['pnl'].mean())) if len(losses) > 0 else None
max_drawdown = (df['equity'].cummax() - df['equity']).max()

# Sharpe & Sortino
returns = df['pnl'] / df['entry']
mean_ret = returns.mean()
std_ret = returns.std()
downside_std = returns[returns < 0].std()

sharpe = (mean_ret / std_ret) * np.sqrt(len(df)) if std_ret > 0 else None
sortino = (mean_ret / downside_std) * np.sqrt(len(df)) if downside_std > 0 else None

# Profit Factor
gross_profit = wins['pnl'].sum()
gross_loss = abs(losses['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

# === Display Metrics ===
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", total_trades)
col2.metric("Win Rate", f"{win_rate:.2f}%")
col3.metric("Avg R:R", f"{avg_rr:.2f}" if avg_rr else "N/A")
col4.metric("Sharpe", f"{sharpe:.2f}" if sharpe else "N/A")
col5.metric("Sortino", f"{sortino:.2f}" if sortino else "N/A")
col6.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor else "N/A")

st.metric("Max Drawdown", f"{max_drawdown:.2f} USDT")

# === Equity Curve ===
fig_equity = px.line(df, x="time", y="equity", title="Equity Curve")
st.plotly_chart(fig_equity, use_container_width=True)

# === Session Performance ===
st.subheader("🕒 Session Performance")
session_perf = df.groupby("session").agg(
    trades=("pnl","count"),
    win_rate=("pnl", lambda x: (x>0).mean()*100),
    avg_pnl=("pnl","mean"),
    total_pnl=("pnl","sum")
).reset_index()

st.dataframe(session_perf)

fig_session = px.bar(session_perf, x="session", y="total_pnl", color="win_rate",
                     title="PnL by Session", text="win_rate")
st.plotly_chart(fig_session, use_container_width=True)

# === Per-Symbol Breakdown ===
st.subheader("📌 Per-Symbol Performance")
symbol_perf = df.groupby("symbol").agg(
    trades=("pnl","count"),
    win_rate=("pnl", lambda x: (x>0).mean()*100),
    avg_pnl=("pnl","mean"),
    total_pnl=("pnl","sum")
).reset_index()

st.dataframe(symbol_perf)

fig_symbol = px.bar(symbol_perf, x="symbol", y="total_pnl", color="win_rate",
                    title="PnL by Symbol", text="win_rate")
st.plotly_chart(fig_symbol, use_container_width=True)

# === Trade History Table ===
st.subheader("📜 Trade History")
st.dataframe(df.tail(20))
from datetime import datetime

def get_session():
    hour = datetime.utcnow().hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "New York"
    else:
        return "Off-hours"
      @app.route('/webhook', methods=['POST'])
def webhook():
    global portfolio_equity, daily_start_equity, open_positions

    data = request.json
    signal = data.get("signal")
    symbol = data.get("symbol", "BTC/USDT")
    price = get_price(symbol)

    # === Session Filter ===
    session = get_session()
    allowed_sessions = ["London", "New York"]  # configure based on analytics
    if session not in allowed_sessions:
        print(f"Signal ignored: {signal} for {symbol} during {session} session")
        return {"status": "ignored", "reason": "outside session"}

    # === Daily Loss Limit ===
    if (portfolio_equity - daily_start_equity) / daily_start_equity <= -daily_loss_limit:
        print("Daily loss limit reached. No more trades today.")
        return {"status": "blocked"}

    # === Max Open Positions ===
    if len(open_positions) >= max_open_positions:
        print("Max open positions reached. Skipping trade.")
        return {"status": "blocked"}

    # === Trade Execution (example stop/target logic) ===
    if signal == "LONG_SIGNAL":
        stop = price * 0.99
        target = price * 1.02
        size = calc_position_size(price, stop, portfolio_equity)
        if size > 0:
            order = exchange.create_market_buy_order(symbol, size)
            open_positions[symbol] = {"side": "long", "entry": price, "stop": stop, "target": target, "size": size}
            print(f"Opened LONG {symbol} @ {price}, stop {stop}, target {target}")

    elif signal == "SHORT_SIGNAL":
        stop = price * 1.01
        target = price * 0.98
        size = calc_position_size(price, stop, portfolio_equity)
        if size > 0:
            order = exchange.create_market_sell_order(symbol, size)
            open_positions[symbol] = {"side": "short", "entry": price, "stop": stop, "target": target, "size": size}
            print(f"Opened SHORT {symbol} @ {price}, stop {stop}, target {target}")

    return {"status": "ok"}  
    import pandas as pd
import datetime

def optimize_sessions(log_file="trade_log.csv"):
    df = pd.read_csv(log_file, names=["time","symbol","side","entry","stop","target","exit","pnl","equity"])
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

    # Map hours to sessions
    def get_session(hour):
        if 0 <= hour < 8:
            return "Asia"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 21:
            return "New York"
        else:
            return "Off-hours"

    df['session'] = df['hour'].apply(get_session)

    # Performance by session
    session_perf = df.groupby("session").agg(
        trades=("pnl","count"),
        win_rate=("pnl", lambda x: (x>0).mean()*100),
        total_pnl=("pnl","sum")
    ).reset_index()

    # Rank sessions by total PnL
    best_sessions = session_perf.sort_values("total_pnl", ascending=False)
    top_sessions = best_sessions[best_sessions['total_pnl'] > 0]['session'].tolist()

    print("Session performance:\n", session_perf)
    print("Optimized allowed sessions:", top_sessions)

    return top_sessions

# Example usage (run weekly, e.g. Sunday night)
allowed_sessions = optimize_sessions()
# Load optimized sessions at startup
allowed_sessions = optimize_sessions()

# Inside webhook:
session = get_session()
if session not in allowed_sessions:
    print(f"Signal ignored: {signal} for {symbol} during {session} session")
    return {"status": "ignored", "reason": "outside session"}
    # pip install optuna backtrader yfinance

import optuna
import backtrader as bt
import yfinance as yf
import pandas as pd

# === Load BTC data ===
data = yf.download("BTC-USD", start="2023-01-01", interval="1h")

# === Backtrader Strategy ===
class ICTStrategy(bt.Strategy):
    params = dict(
        stop_loss=0.01,   # 1%
        take_profit=0.02, # 2%
        rsi_len=14,
        ma_len=50
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_len)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_len)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.ma[0] and self.rsi[0] > 50:
                size = 0.1
                self.buy(size=size)
                self.sell(exectype=bt.Order.Stop, price=self.data.close[0]*(1-self.p.stop_loss))
                self.sell(exectype=bt.Order.Limit, price=self.data.close[0]*(1+self.p.take_profit))
        else:
            pass

# === Objective Function for Optuna ===
def objective(trial):
    stop_loss = trial.suggest_float("stop_loss", 0.005, 0.02)
    take_profit = trial.suggest_float("take_profit", 0.01, 0.05)
    rsi_len = trial.suggest_int("rsi_len", 8, 21)
    ma_len = trial.suggest_int("ma_len", 20, 200)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(ICTStrategy, stop_loss=stop_loss, take_profit=take_profit,
                        rsi_len=rsi_len, ma_len=ma_len)

    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.broker.setcash(10000)
    cerebro.run()

    return cerebro.broker.getvalue()  # maximize final equity

# === Run Optimization ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best Params:", study.best_params)
print("Best Equity:", study.best_value)
# pip install optuna backtrader yfinance

import optuna
import backtrader as bt
import yfinance as yf
import pandas as pd

# === Load BTC data ===
data = yf.download("BTC-USD", start="2023-01-01", interval="1h")

# === Backtrader Strategy ===
class ICTStrategy(bt.Strategy):
    params = dict(
        stop_loss=0.01,   # 1%
        take_profit=0.02, # 2%
        rsi_len=14,
        ma_len=50
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_len)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.ma_len)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.ma[0] and self.rsi[0] > 50:
                size = 0.1
                self.buy(size=size)
                self.sell(exectype=bt.Order.Stop, price=self.data.close[0]*(1-self.p.stop_loss))
                self.sell(exectype=bt.Order.Limit, price=self.data.close[0]*(1+self.p.take_profit))
        else:
            pass

# === Objective Function for Optuna ===
def objective(trial):
    stop_loss = trial.suggest_float("stop_loss", 0.005, 0.02)
    take_profit = trial.suggest_float("take_profit", 0.01, 0.05)
    rsi_len = trial.suggest_int("rsi_len", 8, 21)
    ma_len = trial.suggest_int("ma_len", 20, 200)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(ICTStrategy, stop_loss=stop_loss, take_profit=take_profit,
                        rsi_len=rsi_len, ma_len=ma_len)

    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.broker.setcash(10000)
    cerebro.run()

    return cerebro.broker.getvalue()  # maximize final equity

# === Run Optimization ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best Params:", study.best_params)
import json

# Load optimized parameters
with open("best_params.json", "r") as f:
    best_params = json.load(f)

stop_loss = best_params.get("stop_loss", 0.01)
take_profit = best_params.get("take_profit", 0.02)
rsi_len = best_params.get("rsi_len", 14)
ma_len = best_params.get("ma_len", 50)

print("Loaded optimized params:", best_params)

# Use these values in your trade sizing / stop / target logic
import optuna, backtrader as bt, yfinance as yf, json

# === Load Data ===
data = yf.download("BTC-USD", start="2023-01-01", interval="1h")

# === Strategy Templates ===
class RSIStrategy(bt.Strategy):
    params = dict(rsi_len=14, stop_loss=0.01, take_profit=0.02)
    def __init__(self): self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_len)
    def next(self):
        if not self.position:
            if self.rsi[0] < 30:
                self.buy()
                self.sell(exectype=bt.Order.Stop, price=self.data.close[0]*(1-self.p.stop_loss))
                self.sell(exectype=bt.Order.Limit, price=self.data.close[0]*(1+self.p.take_profit))
            elif self.rsi[0] > 70:
                self.sell()
                self.buy(exectype=bt.Order.Stop, price=self.data.close[0]*(1+self.p.stop_loss))
                self.buy(exectype=bt.Order.Limit, price=self.data.close[0]*(1-self.p.take_profit))

class MAcrossover(bt.Strategy):
    params = dict(short=20, long=50, stop_loss=0.01, take_profit=0.02)
    def __init__(self):
        self.sma1 = bt.indicators.SMA(self.data.close, period=self.p.short)
        self.sma2 = bt.indicators.SMA(self.data.close, period=self.p.long)
    def next(self):
        if not self.position:
            if self.sma1[0] > self.sma2[0]:
                self.buy()
            elif self.sma1[0] < self.sma2[0]:
                self.sell()

# === Objective Function ===
def run_backtest(strategy, **params):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy, **params)
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.broker.setcash(10000)
    cerebro.run()
    return cerebro.broker.getvalue()

def optimize_strategy(strategy, param_space, n_trials=20):
    def objective(trial):
        params = {k: trial.suggest_float(k, *v) if isinstance(v[0], float) else trial.suggest_int(k, *v)
                  for k,v in param_space.items()}
        return run_backtest(strategy, **params)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# === Strategy Candidates ===
strategies = {
    "RSI": (RSIStrategy, {"rsi_len": (8,21), "stop_loss": (0.005,0.02), "take_profit": (0.01,0.05)}),
    "MA": (MAcrossover, {"short": (10,50), "long": (50,200), "stop_loss": (0.005,0.02), "take_profit": (0.01,0.05)})
}

results = {}
for name, (strat, space) in strategies.items():
    best_params, best_value = optimize_strategy(strat, space)
    results[name] = {"params": best_params, "equity": best_value}

# === Select Best Strategy ===
best = max(results.items(), key=lambda x: x[1]["equity"])
print("Best Strategy:", best)

# Save to config
with open("live_config.json","w") as f:
    json.dump({"strategy": best[0], "params": best[1]["params"]}, f)
    class BollingerBreakout(bt.Strategy):
    params = dict(period=20, dev=2, stop_loss=0.01, take_profit=0.02)

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.dev)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.bb.lines.top[0]:
                self.buy()
            elif self.data.close[0] < self.bb.lines.bot[0]:
                self.sell()
              strategies = {
    "RSI": (RSIStrategy, {"rsi_len": (8,21), "stop_loss": (0.005,0.02), "take_profit": (0.01,0.05)}),
    "MA": (MAcrossover, {"short": (10,50), "long": (50,200), "stop_loss": (0.005,0.02), "take_profit": (0.01,0.05)}),
    "Bollinger": (BollingerBreakout, {"period": (10,30), "dev": (1.5,3.0), "stop_loss": (0.005,0.02), "take_profit": (0.01,0.05)})
}  