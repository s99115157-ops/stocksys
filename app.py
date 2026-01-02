import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
from datetime import datetime

# --- 基本環境設定 (移除 Windows 專屬元件) ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI 股市分析網頁版", layout="wide")

# 設定圖表樣式
mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
s = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc)

# --- 核心運算邏輯 (從原始碼移植) ---
def get_data(stock_id, period):
    df = yf.download(stock_id, period=period, interval="1d", progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 計算 KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # 計算 MA
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 買賣訊號
    df['Buy'] = np.where((df['K'] < 30) & (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)), df['Low'] * 0.97, np.nan)
    df['Sell'] = np.where((df['K'] > 70) & (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)), df['High'] * 1.03, np.nan)
    
    return df

# --- 網頁介面 ---
st.title("📈 智慧股市分析指揮中心")
st.info("此版本已優化，支援 Mac 與 iPhone 瀏覽器直接使用")

# 側邊欄
with st.sidebar:
    st.header("查詢設定")
    stock_input = st.text_input("輸入股票代碼", value="2330.TW")
    period_select = st.selectbox("觀測時間長度", ["6mo", "1y", "2y", "3mo"], index=0)
    btn = st.button("執行 AI 分析")

if btn:
    df = get_data(stock_input, period_select)
    
    if df is not None:
        # 準備繪圖元件
        apds = [
            mpf.make_addplot(df['MA5'], color='orange', width=0.8),
            mpf.make_addplot(df['MA20'], color='blue', width=0.8),
            mpf.make_addplot(df['MA60'], color='green', width=0.8),
            mpf.make_addplot(df['Buy'], type='scatter', markersize=100, marker='^', color='red'),
            mpf.make_addplot(df['Sell'], type='scatter', markersize=100, marker='v', color='lime')
        ]
        
        # 顯示看板數字
        c1, c2, c3 = st.columns(3)
        c1.metric("當前股價", f"{df['Close'].iloc[-1]:.2f}")
        c2.metric("K值 / D值", f"{df['K'].iloc[-1]:.1f} / {df['D'].iloc[-1]:.1f}")
        c3.metric("MA20狀態", "站上" if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else "跌破")

        # 顯示圖表
        fig, ax = mpf.plot(df, type='candle', style=s, addplot=apds, 
                           volume=True, returnfig=True, figsize=(12, 8),
                           title=f"\nStock: {stock_input}")
        st.pyplot(fig)
        
        # 顯示原本的文字說明
        st.subheader("📋 策略分析建議")
        st.write(f"1. **KD 訊號**：目前 K 值為 {df['K'].iloc[-1]:.1f}，D 值為 {df['D'].iloc[-1]:.1f}。")
        if df['K'].iloc[-1] < 20: st.warning("⚠️ 目前處於超賣區，請留意打底訊號。")
        if df['K'].iloc[-1] > 80: st.error("⚠️ 目前處於超買區，請留意追高風險。")
        
    else:
        st.error("無法獲取資料，請確認代碼是否輸入正確（台灣股票請加 .TW 或 .TWO）")
