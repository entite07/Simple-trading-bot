import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="NSE AI Trading Bot", layout="wide")

# ======================
# USER INPUT
# ======================
stock = st.text_input("NSE Stock", "RELIANCE.NS")

# ======================
# DOWNLOAD HISTORY
# ======================
data = yf.download(stock, period="6mo", interval="15m")

data["rsi"] = ta.rsi(data["Close"], 14)
data["ema"] = ta.ema(data["Close"], 20)
macd = ta.macd(data["Close"])
data["macd"] = macd["MACD_12_26_9"]

data["future"] = data["Close"].shift(-5) > data["Close"]
data.dropna(inplace=True)

# ======================
# TRAIN AI
# ======================
X = data[["rsi","ema","macd"]]
y = data["future"]
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# ======================
# LIVE DATA
# ======================
live = yf.download(stock, period="1d", interval="5m")
live["rsi"] = ta.rsi(live["Close"], 14)
live["ema"] = ta.ema(live["Close"], 20)
macd = ta.macd(live["Close"])
live["macd"] = macd["MACD_12_26_9"]

last = live.iloc[-1]

# ======================
# SIGNAL
# ======================
def indicator_signal(row):
    if row["rsi"] < 30 and row["Close"] > row["ema"]:
        return "BUY"
    elif row["rsi"] > 70:
        return "SELL"
    else:
        return "HOLD"

signal = indicator_signal(last)

ai_input = np.array([[last["rsi"], last["ema"], last["macd"]]])
ai_pred = model.predict(ai_input)[0]

price = last["Close"]

if signal == "BUY" and ai_pred:
    final = "STRONG BUY"
elif signal == "SELL" and not ai_pred:
    final = "STRONG SELL"
else:
    final = "NO TRADE"

stop_loss = price * 0.98
target = price * 1.04

# ======================
# UI
# ======================
st.title("NSE AI Trading Bot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", round(price,2))
c2.metric("RSI", round(last["rsi"],2))
c3.metric("EMA", round(last["ema"],2))
c4.metric("AI Signal", final)

if final != "NO TRADE":
    st.success(f"Stop Loss: {round(stop_loss,2)} | Target: {round(target,2)}")

# ======================
# CHART
# ======================
fig = go.Figure()
fig.add_trace(go.Scatter(x=live.index, y=live["Close"], name="Price"))
fig.add_trace(go.Scatter(x=live.index, y=live["ema"], name="EMA"))
st.plotly_chart(fig, use_container_width=True)