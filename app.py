import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
from datetime import datetime, timedelta
import urllib3

# --- 環境設定 ---
st.set_page_config(page_title="AI 股市指揮中心", layout="wide")
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 設定繪圖字體 (網頁環境通常使用 DejaVu Sans)
plt.rcParams['axes.unicode_minus'] = False 

# --- 核心邏輯移植 ---
def calculate_indicators(df):
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
    
    # 買賣訊號 (黃箭頭/紫箭頭邏輯)
    df['Buy_Sig'] = np.where((df['K'] < 30) & (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)), df['Low'] * 0.97, np.nan)
    df['Sell_Sig'] = np.where((df['K'] > 70) & (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)), df['High'] * 1.03, np.nan)
    return df

# --- UI 介面 ---
st.title("📈 智慧股市分析系統 (網頁版)")
st.markdown("支援 Mac / iPhone 隨時查看分析結果")

# 側邊欄控制
st.sidebar.header("查詢參數")
target_stock = st.sidebar.text_input("股票代碼 (例: 2330.TW)", value="2330.TW")
period = st.sidebar.selectbox("觀測區間", ["6mo", "1y", "3mo", "5y"], index=0)

if st.sidebar.button("開始分析"):
    with st.spinner('數據計算中...'):
        # 下載數據
        df = yf.download(target_stock, period=period, interval="1d", progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = calculate_indicators(df)
            
            # 準備繪圖
            apds = [
                mpf.make_addplot(df['MA5'], color='#FF9900', width=0.8),
                mpf.make_addplot(df['MA20'], color='#0066FF', width=0.8),
                mpf.make_addplot(df['MA60'], color='#00FF00', width=0.8),
                mpf.make_addplot(df['BB_Up'], color='gray', linestyle='--', alpha=0.3),
                mpf.make_addplot(df['BB_Low'], color='gray', linestyle='--', alpha=0.3),
                mpf.make_addplot(df['Buy_Sig'], type='scatter', markersize=120, marker='^', color='yellow'),
                mpf.make_addplot(df['Sell_Sig'], type='scatter', markersize=120, marker='v', color='purple')
            ]
            
            # 顯示看板資訊
            last_price = df['Close'].iloc[-1]
            last_k = df['K'].iloc[-1]
            last_d = df['D'].iloc[-1]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("當前股價", f"{last_price:.2f}")
            m2.metric("KD 指標", f"K:{last_k:.1f} / D:{last_d:.1f}")
            m3.metric("趨勢", "多頭" if last_price > df['MA20'].iloc[-1] else "空頭")

            # 繪圖
            fig, axlist = mpf.plot(df, type='candle', style='charles',
                                 addplot=apds, volume=True, returnfig=True,
                                 figsize=(12, 7), panel_ratios=(6,2),
                                 title=f"\nStock: {target_stock} Analysis")
            
            st.pyplot(fig)
            
            # 圖例說明
            with st.expander("💡 查看指標圖例"):
                st.write("""
                - **黃色箭頭 (^)**：KD 低檔金叉 (買進參考)
                - **紫色箭頭 (v)**：KD 高檔死叉 (賣出參考)
                - **灰虛線**：布林通道上下軌
                - **線條**：橘色(MA5), 藍色(MA20), 綠色(MA60)
                """)
        else:
            st.error("讀取失敗，請確認代碼是否正確。")