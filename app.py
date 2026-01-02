import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import numpy as np
import warnings
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime, timedelta
import requests
import time
import pytz
import twstock
import concurrent.futures

# --- 頁面設定 ---
st.set_page_config(
    page_title="股票戰略指揮中心 Web版",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 樣式 (維持不變) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { background-color: #4a90e2; color: white; border-radius: 5px; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #357abd; }
    div[data-testid="stMetricLabel"] { color: #b0b0b0 !important; font-size: 14px; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #FFD700 !important; font-family: 'Microsoft JhengHei', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 環境設定 ---
warnings.filterwarnings("ignore")

# --- 0. 股票清單與快取 (維持原樣) ---
DEFAULT_STOCK_MAP = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電', '2382': '廣達', 
    '2881': '富邦金', '2882': '國泰金', '2603': '長榮', '2609': '陽明', '2615': '萬海',
    '3231': '緯創', '3037': '欣興', '2379': '瑞昱', '3034': '聯詠', '2303': '聯電',
    '8069': '元太', '5347': '世界', '3293': '鈊象', '3105': '穩懋', '6488': '環球晶',
    '3324': '雙鴻', '3017': '奇鋐', '6547': '高端疫苗', '6550': '北極星藥業',
    '2356': '英業達', '2376': '技嘉', '3443': '創意', '3661': '世芯-KY', '2327': '國巨',
    '2891': '中信金', '2886': '兆豐金', '5880': '合庫金', '2884': '玉山金', '1605': '華新'
}

ALL_STOCKS_CACHE = {}
STOCK_SUFFIX_MAP = {}

@st.cache_data(ttl=86400) 
def fetch_tw_stock_list():
    global ALL_STOCKS_CACHE, STOCK_SUFFIX_MAP
    if ALL_STOCKS_CACHE: return ALL_STOCKS_CACHE, STOCK_SUFFIX_MAP

    stock_dict = DEFAULT_STOCK_MAP.copy()
    local_suffix_map = {}

    for code in stock_dict:
        local_suffix_map[code] = '.TW'
        if code in ['8069', '5347', '3293', '3105', '6488', '3324', '8039', '6274', '8358', '1341', '3017', '3607', '2457']:
            local_suffix_map[code] = '.TWO'
    
    local_suffix_map['3017'] = '.TW' 
    local_suffix_map['3324'] = '.TWO'

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        # 上市
        try:
            url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
            res = requests.get(url_twse, verify=False, headers=headers, timeout=2)
            if res.status_code == 200:
                df = pd.read_html(res.text)[0].iloc[2:]
                for item in df[0]:
                    parts = str(item).split()
                    if len(parts) >= 2:
                        stock_dict[parts[0]] = parts[1]
                        local_suffix_map[parts[0]] = '.TW'
        except: pass

        # 上櫃
        try:
            url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
            res = requests.get(url_tpex, verify=False, headers=headers, timeout=2)
            if res.status_code == 200:
                df = pd.read_html(res.text)[0].iloc[2:]
                for item in df[0]:
                    parts = str(item).split()
                    if len(parts) >= 2:
                        stock_dict[parts[0]] = parts[1]
                        local_suffix_map[parts[0]] = '.TWO'
        except: pass
            
        ALL_STOCKS_CACHE = stock_dict
        STOCK_SUFFIX_MAP = local_suffix_map
        return stock_dict, local_suffix_map

    except:
        return DEFAULT_STOCK_MAP, local_suffix_map

def get_stock_info(code):
    stock_map, suffix_map = fetch_tw_stock_list()
    pure_code = code.split('.')[0]
    name = stock_map.get(pure_code, pure_code)
    suffix = suffix_map.get(pure_code, '.TW')
    return pure_code, name, suffix

def get_realtime_data(pure_code, suffix):
    try:
        is_otc = (suffix == '.TWO')
        if is_otc:
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                ts = int(time.time() * 1000)
                url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=otc_{pure_code}.tw&json=1&delay=0&_={ts}"
                res = requests.get(url, headers=headers, timeout=3, verify=False)
                data = res.json()
                if 'msgArray' in data and len(data['msgArray']) > 0:
                    info = data['msgArray'][0]
                    price_cand = [info.get('z', '-'), info.get('b', '0').split('_')[0], info.get('a', '0').split('_')[0], info.get('y', '0')]
                    final_price = 0.0
                    for p in price_cand:
                        if p != '-' and float(p) > 0:
                            final_price = float(p)
                            break
                    vol = float(info.get('v', 0))
                    return {'price': final_price, 'volume': vol, 'high': float(info.get('h', final_price)), 'low': float(info.get('l', final_price))}
            except: pass
        else:
            stock = twstock.realtime.get(pure_code)
            if stock['success']:
                info = stock['realtime']
                price = info.get('latest_trade_price', None)
                if not price or price == '-': price = info.get('best_bid_price', [0])[0]
                return {
                    'price': float(price),
                    'volume': float(info.get('accumulate_trade_volume', 0)),
                    'high': float(info.get('high', price)),
                    'low': float(info.get('low', price))
                }
    except: pass
    return None

def calculate_trend_line(df, period=20):
    if len(df) < period: return pd.Series([np.nan]*len(df), index=df.index)
    trend_line = [np.nan] * len(df)
    y_data = df['Close'].values
    x_data = np.arange(len(df))
    start_idx = len(df) - period * 2
    for i in range(max(period, start_idx), len(df)):
        y_slice = y_data[i-period:i]
        x_slice = x_data[i-period:i]
        slope, intercept, _, _, _ = linregress(x_slice, y_slice)
        trend_line[i] = slope * i + intercept
    return pd.Series(trend_line, index=df.index)

def process_indicators(df, aggressive_mode=False):
    df = df.copy()
    for d in [5, 10, 20, 60, 120]: df[f'MA{d}'] = df['Close'].rolling(window=d).mean()
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['MA20'] + 2 * df['Std']
    df['BB_Low'] = df['MA20'] - 2 * df['Std']
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['TrendLine'] = calculate_trend_line(df)

    buy_signals, sell_signals, div_buy, ma_sup, double_gold_signals = [], [], [], [], []
    last_double_gold_idx = -99
    
    for i in range(len(df)):
        if i < 60:
            for l in [buy_signals, sell_signals, div_buy, ma_sup, double_gold_signals]: l.append(np.nan)
            continue
            
        k, d = df['K'].iloc[i], df['D'].iloc[i]
        prev_k, prev_d = df['K'].iloc[i-1], df['D'].iloc[i-1]
        is_gold_cross = (prev_k < prev_d) and (k > d)
        is_dead_cross = (prev_k > prev_d) and (k < d)
        ma20_curr = df['MA20'].iloc[i]
        ma20_prev = df['MA20'].iloc[i-1] if i > 0 else ma20_curr
        is_ma20_rising = (ma20_curr >= ma20_prev)
        
        basic_buy = is_gold_cross and (k < 80) if aggressive_mode else is_gold_cross and (k < 80) and is_ma20_rising
        
        is_double_gold = False
        # [核心修復] 嚴格的二次金叉邏輯 (完全參照 Stock_Final.py)
        if is_gold_cross and (k < 60) and (i - last_double_gold_idx > 8) and (df['Hist'].iloc[i] > df['Hist'].iloc[i-1]):
            # 必須站上月線 (除非是搶反彈模式)
            trend_check = True if aggressive_mode else (df['Close'].iloc[i] > ma20_curr)
            
            if trend_check:
                # 回溯尋找前一次金叉
                for j in range(8, 30):
                    idx = i - j
                    if idx < 0: break
                    past_k = df['K'].iloc[idx]; past_d = df['D'].iloc[idx]
                    past_prev_k = df['K'].iloc[idx-1]; past_prev_d = df['D'].iloc[idx-1]
                    
                    past_is_gold = (past_prev_k < past_prev_d) and (past_k > past_d)
                    
                    if past_is_gold:
                        # 檢查兩次金叉中間是否有回檔 (中間的 K 值最低點必須夠低，形成N字)
                        mid_period_k = df['K'].iloc[idx:i]
                        min_k_in_between = mid_period_k.min()
                        
                        # 這次金叉的位置比上次高 (多頭排列) 且 中間有回測
                        if (k > past_k) and (min_k_in_between < past_k * 0.9):
                            is_double_gold = True
                            last_double_gold_idx = i
                            break
        
        buy_signals.append(df['Low'].iloc[i] * 0.99 if basic_buy else np.nan)
        sell_signals.append(df['High'].iloc[i] * 1.01 if is_dead_cross else np.nan)
        double_gold_signals.append(df['Low'].iloc[i]*0.98 if is_double_gold else np.nan)
        
        is_div = (df['K'].iloc[i-1] < df['K'].iloc[i-2]) and (df['K'].iloc[i-1] < k) and (df['Low'].iloc[i] < df['Low'].iloc[i-1]) and (k > df['K'].iloc[i-1]) and (k < 40)
        div_buy.append(df['Low'].iloc[i]*0.99 if is_div else np.nan)
        
        is_ma_sup = (not np.isnan(df['MA60'].iloc[i])) and (df['Low'].iloc[i] <= df['MA60'].iloc[i] * 1.02) and (df['Close'].iloc[i] >= df['MA60'].iloc[i])
        ma_sup.append(df['MA60'].iloc[i] if is_ma_sup else np.nan)

    df['Buy'] = buy_signals
    df['Sell'] = sell_signals
    df['Div_Buy'] = div_buy
    df['MA60_Sup'] = ma_sup
    df['Double_Gold'] = double_gold_signals
    return df

def detect_w_bottom(df):
    w_lines = []
    min_idxs = argrelextrema(df['Low'].values, np.less, order=5)[0]
    if len(min_idxs) < 2: return w_lines
    recent_mins = [i for i in min_idxs if i > len(df) - 100]
    if len(recent_mins) < 2: return w_lines
    for i in range(len(recent_mins)-1, 0, -1):
        idx1, idx2 = recent_mins[i-1], recent_mins[i]
        if idx2 - idx1 > 50: continue
        mid_slice = df['High'].iloc[idx1:idx2]
        if len(mid_slice) == 0: continue
        idx_max = mid_slice.idxmax()
        price_max = df['High'].loc[idx_max]
        price1 = df['Low'].iloc[idx1]
        price2 = df['Low'].iloc[idx2]
        if 0.95 * price1 <= price2 <= 1.05 * price1:
            w_lines.append([(df.index[idx1], price1), (idx_max, price_max), (df.index[idx2], price2)])
            w_lines.append([(df.index[idx1], price_max), (df.index[-1], price_max)])
            break
    return w_lines

# --- [重寫] 智能選股邏輯 (完全移植 Stock_Final.py) ---
def scan_single_stock_full_logic(code, name, only_double):
    """
    這是從 Stock_Final.py 移植的完整版邏輯
    包含 AI 評分系統與完整的訊號過濾
    """
    try:
        suffix = STOCK_SUFFIX_MAP.get(code, '.TW')
        ticker = f"{code}{suffix}"
        
        # 下載資料 (至少需要 60 天來計算指標)
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
        
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 確保成交量不是 0
        if df['Volume'].iloc[-1] == 0: return None

        # 計算必要指標 (在 thread 內部獨立計算)
        # 為了效能，這裡手動快速算，不呼叫 process_indicators
        close = df['Close']
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        # --- AI 評分系統 (Stock_Final.py 核心) ---
        ai_score = 50
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        if latest['Close'] > ma20.iloc[-1]: ai_score += 15
        if ma20.iloc[-1] > ma20.iloc[-2]: ai_score += 10
        if k.iloc[-1] > k.iloc[-2]: ai_score += 10
        if latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1]: ai_score += 10
        if latest['Close'] > ma60.iloc[-1]: ai_score += 5

        signals = []
        
        # --- 二次金叉判斷 ---
        is_double_gold = False
        cur_k = k.iloc[-1]; cur_d = d.iloc[-1]
        prev_k = k.iloc[-2]; prev_d = d.iloc[-2]
        
        macd_momentum = hist.iloc[-1] > hist.iloc[-2]
        
        # KD金叉 且 位階不高 且 MACD 轉強
        if (prev_k < prev_d) and (cur_k > cur_d) and (cur_k < 60) and macd_momentum:
            if latest['Close'] > ma20.iloc[-1]:
                # 回溯迴圈
                for j in range(8, 30):
                    if len(k) < j+2: break
                    past_k = k.iloc[-1-j]; past_d = d.iloc[-1-j]
                    past_prev_k = k.iloc[-1-j-1]; past_prev_d = d.iloc[-1-j-1]
                    
                    past_is_gold = (past_prev_k < past_prev_d) and (past_k > past_d)
                    
                    if past_is_gold:
                        mid_period_k = k.iloc[-1-j:-1]
                        min_k_in_between = mid_period_k.min()
                        
                        if (cur_k > past_k) and (min_k_in_between < past_k * 0.9):
                            is_double_gold = True
                            break
        
        if is_double_gold:
            signals.append("★二次金叉")
            ai_score += 30 

        # 如果使用者只要二次金叉，且這檔不是，就直接跳過
        if only_double and not is_double_gold:
            return None 

        # 其他強勢訊號
        if latest['Close'] > ma20.iloc[-1] and ma20.iloc[-1] > ma20.iloc[-2]:
            pct = (latest['Close'] - prev['Close']) / prev['Close'] * 100
            if pct > 2: signals.append("強勢上攻")

        if k.iloc[-2] < 30 and k.iloc[-1] > k.iloc[-2]:
            signals.append("KD低檔金叉")
        
        if prev['Close'] < ma60.iloc[-2] and latest['Close'] > ma60.iloc[-1]:
            signals.append("突破季線")

        # 最終過濾：必須要有訊號 且 分數及格 (原本是 >=60)
        if signals and ai_score >= 60:
            change_p = (latest['Close'] - prev['Close']) / prev['Close'] * 100
            return {
                "代號": code,
                "名稱": name,
                "現價": f"{latest['Close']:.2f}",
                "漲跌": f"{change_p:.2f}%",
                "訊號": " ".join(signals),
                "AI分": ai_score
            }
        
    except Exception:
        return None
    return None

# --- UI 佈局 ---

st.sidebar.title("🎮 戰略指揮中心")
stock_input = st.sidebar.text_input("輸入股票代號", value="2330", max_chars=8)
pure_code, stock_name, suffix = get_stock_info(stock_input)

st.sidebar.subheader("時間週期")
interval_map = {"日線": "1d", "60分": "60m", "30分": "30m", "週線": "1wk", "月線": "1mo"}
selected_interval_label = st.sidebar.radio("選擇週期", list(interval_map.keys()), horizontal=True, label_visibility="collapsed")
selected_interval = interval_map[selected_interval_label]

st.sidebar.subheader("圖表疊加")
col_chk1, col_chk2 = st.sidebar.columns(2)
with col_chk1:
    show_ma = st.checkbox("均線 (MA)", value=True)
    show_trend = st.checkbox("趨勢線", value=True)
with col_chk2:
    show_bb = st.checkbox("布林 (BB)", value=True)
    show_pattern = st.checkbox("型態", value=True)

aggressive = st.sidebar.checkbox("🐟 搶反彈模式 (敏銳)", value=False)
refresh_btn = st.sidebar.button("🔄 立即刷新數據", use_container_width=True)
if refresh_btn: st.cache_data.clear()

st.sidebar.markdown("---")

# --- 側邊欄掃描器 (已修復) ---
with st.sidebar.expander("🔍 智能選股掃描", expanded=True):
    # [新增] 掃描深度選擇，解決 "掃不到股票" 的問題
    scan_limit = st.select_slider(
        "掃描範圍 (檔數)", 
        options=[50, 100, 200, 500],
        value=100
    )
    st.caption(f"將從資料庫中掃描前 {scan_limit} 檔熱門股")
    
    only_double = st.checkbox("只顯示二次金叉 (大噴發)", value=False)
    
    if st.button("🚀 開始渦輪掃描", use_container_width=True):
        stock_list, _ = fetch_tw_stock_list()
        scan_results = []
        
        # 取得要掃描的代號列表
        all_codes = list(stock_list.keys())
        # 如果快取還沒抓到全部，先用預設的，否則用全部的前 N 檔
        target_codes = all_codes[:scan_limit]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(target_codes)
        
        # 使用 ThreadPool 進行併發掃描 (Max workers 設高一點以加快 IO)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交任務
            futures = {executor.submit(scan_single_stock_full_logic, c, stock_list[c], only_double): c for c in target_codes}
            
            done_count = 0
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                if result:
                    scan_results.append(result)
                
                done_count += 1
                # 更新進度條
                progress_bar.progress(done_count / total)
                status_text.text(f"掃描進度: {done_count}/{total}")
        
        progress_bar.empty()
        status_text.empty()
        
        if scan_results:
            st.success(f"掃描完成！發現 {len(scan_results)} 檔潛力股")
            df_res = pd.DataFrame(scan_results).sort_values(by="AI分", ascending=False)
            st.dataframe(
                df_res, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "現價": st.column_config.TextColumn("現價"),
                    "漲跌": st.column_config.TextColumn("漲跌"),
                }
            )
        else:
            st.warning("⚠️ 範圍內無符合強勢條件的股票，請嘗試擴大掃描範圍或取消「只顯示二次金叉」。")

# --- 主畫面邏輯 ---
full_ticker = f"{pure_code}{suffix}"

try:
    with st.spinner(f"正在連線 {full_ticker} ..."):
        period = "1mo" if selected_interval in ["60m", "30m"] else "max" if selected_interval == "1mo" else "2y"
        df = yf.download(full_ticker, period=period, interval=selected_interval, progress=False)
        
        if df.empty:
            st.error("無法取得資料，請確認代號。")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if selected_interval == '1d':
            rt_data = get_realtime_data(pure_code, suffix)
            if rt_data and rt_data['price'] > 0:
                current_price = rt_data['price']
                current_vol = rt_data['volume'] * 1000 
                last_idx = df.index[-1]
                df.loc[last_idx, 'Close'] = current_price
                df.loc[last_idx, 'High'] = max(df.loc[last_idx, 'High'], rt_data['high'])
                df.loc[last_idx, 'Low'] = min(df.loc[last_idx, 'Low'], rt_data['low'])
                df.loc[last_idx, 'Volume'] = current_vol

        df = process_indicators(df, aggressive)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# 顯示數據卡片
last_row = df.iloc[-1]
prev_row = df.iloc[-2]
change = last_row['Close'] - prev_row['Close']
pct = (change / prev_row['Close']) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{stock_name} ({pure_code})", f"{last_row['Close']:.2f}", f"{change:.2f} ({pct:.2f}%)")
vol_display = int(last_row['Volume'] / 1000) 
c2.metric("成交量", f"{vol_display:,} 張")
c3.metric("KD 指標", f"K{last_row['K']:.1f} / D{last_row['D']:.1f}", "黃金交叉" if last_row['K'] > last_row['D'] else "死亡交叉", delta_color="off")

ai_score = 50
if last_row['Close'] > last_row['MA20']: ai_score += 10
if not np.isnan(last_row['Double_Gold']): ai_score = 100
ai_msg = "🔥 強力看多" if ai_score >= 80 else "📈 偏多操作" if ai_score >= 60 else "🐻 觀望/偏空"
c4.metric("AI 訊號", ai_msg, f"{ai_score}分")

# 繪圖
plot_df = df.tail(100)
addplots = []

if show_ma:
    if not plot_df['MA5'].isnull().all(): addplots.append(mpf.make_addplot(plot_df['MA5'], color='#FFFF00', width=1.0))
    if not plot_df['MA20'].isnull().all(): addplots.append(mpf.make_addplot(plot_df['MA20'], color='#CC00FF', width=1.2))
    if not plot_df['MA60'].isnull().all(): addplots.append(mpf.make_addplot(plot_df['MA60'], color='#00FF00', width=1.2))

if show_bb:
    addplots.append(mpf.make_addplot(plot_df['BB_Up'], color='gray', linestyle='--', width=0.8))
    addplots.append(mpf.make_addplot(plot_df['BB_Low'], color='gray', linestyle='--', width=0.8))

if show_trend:
    addplots.append(mpf.make_addplot(plot_df['TrendLine'], color='cyan', linestyle='--', width=1.0))

if not plot_df['Buy'].isnull().all(): 
    addplots.append(mpf.make_addplot(plot_df['Buy'], type='scatter', markersize=60, marker='^', color='#FF1744'))
if not plot_df['Sell'].isnull().all(): 
    addplots.append(mpf.make_addplot(plot_df['Sell'], type='scatter', markersize=60, marker='v', color='#00FF00'))
if not plot_df['Double_Gold'].isnull().all():
    addplots.append(mpf.make_addplot(plot_df['Double_Gold'], type='scatter', markersize=150, marker='*', color='red'))

addplots.append(mpf.make_addplot(plot_df['K'], panel=2, color='#ffa726', width=1, ylabel='KD'))
addplots.append(mpf.make_addplot(plot_df['D'], panel=2, color='#42a5f5', width=1))
macd_colors = ['#ef5350' if v >= 0 else '#26a69a' for v in plot_df['Hist']]
addplots.append(mpf.make_addplot(plot_df['Hist'], type='bar', panel=3, color=macd_colors, ylabel='MACD'))

my_style = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(up='#ef5350', down='#26a69a', edge='inherit', wick='inherit', volume={'up':'#ef5350','down':'#26a69a'}),
    base_mpf_style='nightclouds',
    gridstyle=':',
    rc={'font.family': 'Sans-Serif', 'axes.labelsize': 8}
)

x_max = len(plot_df) + 8
fig, axlist = mpf.plot(
    plot_df,
    type='candle',
    volume=True,
    addplot=addplots,
    style=my_style,
    returnfig=True,
    panel_ratios=(6, 2, 2, 2),
    figratio=(16, 9),
    figscale=1.2,
    datetime_format='%m-%d',
    xlim=(0, x_max),
    tight_layout=True
)

st.pyplot(fig)
st.caption("Designed by Gemini AI Partner | Stock Command Center v2.2 (Final Engine)")
