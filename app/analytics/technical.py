import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import Dict, Any

class TechnicalAnalysisEngine:
    """
    Calculates technical indicators and momentum signals using market data.
    """

    def __init__(self):
        pass

    def fetch_and_analyze(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetches historical price data and computes SMA, EMA, RSI, MACD, and Bollinger Bands.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty or len(df) < 30:
            return {
                "ticker": ticker.upper(),
                "error": f"Insufficient price data available for ticker '{ticker}'."
            }

        # Calculate Moving Averages
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
        df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

        # Calculate Momentum Indicators
        df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
        
        # Calculate MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()

        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_High"] = bollinger.bollinger_hband()
        df["BB_Low"] = bollinger.bollinger_lband()

        # Latest Indicator Values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = round(float(latest["Close"]), 2)
        prev_price = round(float(prev["Close"]), 2)
        price_change_pct = round(((current_price - prev_price) / prev_price) * 100, 2)
        
        rsi_val = round(float(latest["RSI_14"]), 2) if not np.isnan(latest["RSI_14"]) else None
        sma_50 = round(float(latest["SMA_50"]), 2) if not np.isnan(latest["SMA_50"]) else None
        sma_200 = round(float(latest["SMA_200"]), 2) if not np.isnan(latest["SMA_200"]) else None
        macd_val = round(float(latest["MACD"]), 2) if not np.isnan(latest["MACD"]) else None
        macd_sig = round(float(latest["MACD_Signal"]), 2) if not np.isnan(latest["MACD_Signal"]) else None

        # Signal Synthesis
        signals = []
        if rsi_val:
            if rsi_val > 70:
                signals.append("RSI indicates Overbought conditions (>70).")
            elif rsi_val < 30:
                signals.append("RSI indicates Oversold conditions (<30).")
            else:
                signals.append(f"RSI is Neutral at {rsi_val}.")

        if sma_50 and current_price > sma_50:
            signals.append("Trading above 50-day SMA (Bullish intermediate trend).")
        elif sma_50:
            signals.append("Trading below 50-day SMA (Bearish intermediate trend).")

        if sma_50 and sma_200:
            if sma_50 > sma_200:
                signals.append("Golden Cross active (50-day SMA > 200-day SMA).")
            else:
                signals.append("Death Cross active (50-day SMA < 200-day SMA).")

        if macd_val and macd_sig:
            if macd_val > macd_sig:
                signals.append("MACD is above Signal Line (Bullish momentum).")
            else:
                signals.append("MACD is below Signal Line (Bearish momentum).")

        # Convert history to serializable records for charts
        history_records = []
        for date_idx, row in df.tail(120).iterrows(): # Last 120 trading days
            history_records.append({
                "Date": date_idx.strftime("%Y-%m-%d"),
                "Open": round(float(row["Open"]), 2),
                "High": round(row["High"], 2),
                "Low": round(row["Low"], 2),
                "Close": round(row["Close"], 2),
                "Volume": int(row["Volume"]),
                "SMA_50": round(float(row["SMA_50"]), 2) if not np.isnan(row["SMA_50"]) else None,
                "SMA_200": round(float(row["SMA_200"]), 2) if not np.isnan(row["SMA_200"]) else None,
                "RSI_14": round(float(row["RSI_14"]), 2) if not np.isnan(row["RSI_14"]) else None,
            })

        return {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "rsi_14": rsi_val,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "macd": macd_val,
            "macd_signal": macd_sig,
            "technical_signals": signals,
            "history": history_records
        }