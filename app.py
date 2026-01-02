import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import warnings
from datetime import datetime, timedelta
import requests
import urllib3

# --- 基本設定 ---
st.set_page_config(page_title="AI 股市指揮中心", layout="wide")
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 解決中文亂碼 (Streamlit Cloud 需載入字體，這裡預設使用支援的中文字體名稱)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 資料下載與計算核心 (從原始碼移植) ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    
    # 計算指標 (KD, MA, Bollinger)
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    
    # KD 計算
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # 布林通道
    std = close.rolling(20).std()
    df['BB_Up'] = df['MA20'] + 2 * std
    df['BB_Low'] = df['MA20'] - 2 * std
    
    # 買賣訊號邏輯 (簡化移植)
    df['Buy'] = np.where((df['K'] < 30) & (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)), df['Low'] * 0.98, np.nan)
    df['Sell'] = np.where((df['K'] > 70) & (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)), df['High'] * 1.02, np.nan)
    
    return df

# --- UI 介面設計 ---
st.sidebar.title("📈 參數設定")
stock_id = st.sidebar.text_input("輸入股票代碼 (例: 2330.TW)", value="2330.TW")
time_range = st.sidebar.selectbox("時段", ["6mo", "1y", "1mo", "5d"], index=0)
interval = st.sidebar.selectbox("頻率", ["1d", "60m", "30m", "1wk"], index=0)

if st.sidebar.button("開始分析"):
    with st.spinner('正在分析數據...'):
        df = fetch_stock_data(stock_id, time_range, interval)
        
        if df is not None:
            # 建立圖表
            apds = [
                mpf.make_addplot(df['MA5'], color='orange', width=0.7),
                mpf.make_addplot(df['MA20'], color='blue', width=0.7),
                mpf.make_addplot(df['MA60'], color='green', width=0.7),
                mpf.make_addplot(df['Buy'], type='scatter', markersize=100, marker='^', color='red'),
                mpf.make_addplot(df['Sell'], type='scatter', markersize=100, marker='v', color='lime')
            ]
            
            fig, axlist = mpf.plot(df, type='candle', style='charles', 
                                   addplot=apds, returnfig=True, 
                                   figsize=(12, 8), volume=True,
                                   title=f"\nStock: {stock_id}")
            
            # 顯示資訊欄位
            col1, col2, col3 = st.columns(3)
            latest = df.iloc[-1]
            col1.metric("當前價格", f"{latest['Close']:.2f}")
            col2.metric("K / D 值", f"{latest['K']:.1f} / {latest['D']:.1f}")
            col3.metric("MA20 趨勢", "↑ 偏多" if latest['Close'] > latest['MA20'] else "↓ 偏空")
            
            # 渲染圖表
            st.pyplot(fig)
            
            # 顯示說明
            st.markdown("""
            ### 💡 圖例說明
            * **紅箭頭 (^)**：KD 低檔金叉買進訊號
            * **綠箭頭 (v)**：KD 高檔死叉賣出訊號
            * **線條**：橘(MA5), 藍(MA20), 綠(MA60)
            """)
        else:
            st.error("找不到該股票代碼，請確認後綴是否正確 (如 .TW 或 .TWO)")

# 智能選股區塊 (移植原本的 SmartScreener 概念)
with st.expander("🔍 快速選股掃描"):
    if st.button("啟動 AI 強勢股掃描"):
        st.write("掃描功能運行中... (範例顯示)")
        # 這裡可以放置你原本 ThreadPoolExecutor 的邏輯