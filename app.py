import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Shaktimaan Hybrid Scanner", layout="wide")
st.title("📊 Shaktimaan Hybrid — Multi-Stock Classification & Stop Loss")

# ==========================================
# 1. WATCHLIST INPUT & SETTINGS
# ==========================================
col1, col2 = st.columns([2, 1])
with col1:
    default_tickers = (
        "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, BHARTIARTL.NS, "
        "ICICIBANK.NS, SBIN.NS, LT.NS, AAPL, MSFT, NVDA, AZN.L, SHEL.L"
    )
    ticker_input = st.text_area(
        "Paste Stock Tickers (comma or new-line separated — use .NS for NSE, .BO for BSE, .L for UK)",
        default_tickers,
        height=110,
    )

with col2:
    uploaded_file = st.file_uploader("Or upload a CSV / TXT watchlist", type=["csv", "txt"])
    timeframe_sl = st.selectbox(
        "Active Chart / Stop Loss Reference",
        ["Weekly (13 EMA)", "Monthly (20 EMA)", "Daily (13 EMA)"],
        index=0,
    )
    vol_surge_mult = st.number_input("Satellite Volume Surge Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Parse and clean ticker list
raw_text = uploaded_file.read().decode("utf-8") if uploaded_file is not None else ticker_input
tickers = list(dict.fromkeys([t.strip().upper() for t in raw_text.replace("\n", ",").split(",") if t.strip()]))

# ==========================================
# 2. TECHNICAL ENGINE (PINE SCRIPT REPLICA)
# ==========================================
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vstop(close: pd.Series, high: pd.Series, low: pd.Series, length: int = 20, mult: float = 2.5):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, min_periods=length).mean()
    vdist = atr * mult

    c_vals = close.values
    d_vals = vdist.values
    vstop = np.full(len(c_vals), np.nan)
    vtrend = np.ones(len(c_vals), dtype=int)

    for i in range(1, len(c_vals)):
        if np.isnan(d_vals[i]):
            continue
        if np.isnan(vstop[i - 1]):
            vtrend[i] = 1
            vstop[i] = c_vals[i] - d_vals[i]
        else:
            if vtrend[i - 1] == 1:
                bull_stop = max(vstop[i - 1], c_vals[i] - d_vals[i])
                if c_vals[i] < vstop[i - 1]:
                    vtrend[i] = -1
                    vstop[i] = c_vals[i] + d_vals[i]
                else:
                    vtrend[i] = 1
                    vstop[i] = bull_stop
            else:
                bear_stop = min(vstop[i - 1], c_vals[i] + d_vals[i])
                if c_vals[i] > vstop[i - 1]:
                    vtrend[i] = 1
                    vstop[i] = c_vals[i] - d_vals[i]
                else:
                    vtrend[i] = -1
                    vstop[i] = bear_stop
    return vstop[-1], vtrend[-1]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_analyze_batch(ticker_tuple: tuple, sl_mode: str, vol_mult: float) -> pd.DataFrame:
    ticker_list = list(ticker_tuple)
    results = []

    # Download in chunks of 40 to prevent Yahoo Finance rate limits
    chunk_size = 40
    for i in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[i : i + chunk_size]
        raw_data = yf.download(
            tickers=chunk,
            period="6y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if raw_data.empty:
            continue

        for t in chunk:
            try:
                if len(chunk) == 1:
                    df_d = raw_data.copy()
                    if isinstance(df_d.columns, pd.MultiIndex):
                        df_d.columns = df_d.columns.get_level_values(0)
                else:
                    if t not in raw_data.columns.get_level_values(0):
                        continue
                    df_d = raw_data[t].dropna(how="all").copy()

                df_d = df_d.dropna(subset=["Close", "High", "Low", "Volume"])
                if len(df_d) < 252:
                    continue

                # Resample to Weekly (W-FRI) and Monthly (ME)
                agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                df_w = df_d.resample("W-FRI").agg(agg_rules).dropna()
                df_m = df_d.resample("ME").agg(agg_rules).dropna()

                if len(df_m) < 61 or len(df_w) < 53:
                    continue

                curr_close = float(df_d["Close"].iloc[-1])

                # --- 1. CORE (Strictly Monthly "M") ---
                m_close = float(df_m["Close"].iloc[-1])
                m_sma20 = float(df_m["Close"].rolling(20).mean().iloc[-1])
                m_sma60 = float(df_m["Close"].rolling(60).mean().iloc[-1])
                m_close12 = float(df_m["Close"].iloc[-13])
                m_roc12 = ((m_close - m_close12) / m_close12) * 100.0
                tf_core = (m_close > m_sma60) and (m_sma20 > m_sma60) and (m_roc12 > 15.0)

                # --- 2. SATELLITE (Strictly Weekly "W") ---
                w_close = float(df_w["Close"].iloc[-1])
                w_vol = float(df_w["Volume"].iloc[-1])
                w_vol_sma20 = float(df_w["Volume"].rolling(20).mean().iloc[-1])
                w_high = float(df_w["High"].iloc[-1])
                w_high52 = float(df_w["High"].iloc[-53:-1].max())
                w_ema21 = float(df_w["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
                w_close13 = float(df_w["Close"].iloc[-14])
                w_roc13 = ((w_close - w_close13) / w_close13) * 100.0

                w_vol_spike = w_vol > (w_vol_sma20 * vol_mult)
                w_52w_high = w_high >= w_high52
                tf_satellite = (w_roc13 > 25.0) or w_vol_spike or (w_close > w_ema21 and not tf_core)

                if tf_core and tf_satellite:
                    classification = "CORE+SATELLITE"
                elif tf_core:
                    classification = "CORE"
                elif tf_satellite:
                    classification = "SATELLITE"
                else:
                    classification = "Neither"

                # --- 3. WEEKLY STAGE ENGINE ---
                w_ema20_s = df_w["Close"].ewm(span=20, adjust=False).mean()
                w_ema20 = float(w_ema20_s.iloc[-1])
                w_ema20_past = float(w_ema20_s.iloc[-6])
                w_slope = (w_ema20 - w_ema20_past) / w_ema20_past

                if w_close > w_ema20:
                    stage = "Stage 1: Basing" if w_slope < -0.0005 else "Stage 2: Advancing"
                else:
                    stage = "Stage 3: Distribution" if w_slope >= -0.0005 else "Stage 4: Declining"

                # --- 4. BREAKOUT TRIGGERS (DC / 52W / BB(50,2)) ---
                df_ref = df_w if "Weekly" in sl_mode else (df_m if "Monthly" in sl_mode else df_d)
                target_bars = 52 if "Weekly" in sl_mode else (12 if "Monthly" in sl_mode else 252)

                ref_high = float(df_ref["High"].iloc[-1])
                ref_close = float(df_ref["Close"].iloc[-1])

                is_52w = ref_high >= float(df_ref["High"].iloc[-(target_bars + 1) : -1].max())
                is_dc = ref_high > float(df_ref["High"].iloc[-21:-1].max())

                bb_basis = df_ref["Close"].rolling(50).mean().iloc[-1]
                bb_std = df_ref["Close"].rolling(50).std().iloc[-1]
                is_bb = ref_close > (bb_basis + 2.0 * bb_std)

                dc_col = "DC" if is_dc else "-"
                w52_col = "52W" if is_52w else "-"
                bb_col = "BB(50,2)" if is_bb else "-"

                # --- 5. ABC / XYZ REGIME & ACTION ---
                rsi_s = compute_rsi(df_ref["Close"], 14)
                rsi_ma_s = rsi_s.rolling(14).mean()
                curr_rsi = float(rsi_s.iloc[-1])
                curr_rsi_ma = float(rsi_ma_s.iloc[-1])
                _, vtrend_last = compute_vstop(df_ref["Close"], df_ref["High"], df_ref["Low"])

                if curr_rsi >= 55:
                    if curr_rsi >= curr_rsi_ma:
                        abc = 3 if curr_rsi > 70 else (2 if curr_rsi >= 60 else 1)
                    else:
                        abc = 3 if curr_rsi > 74 else (2 if curr_rsi >= 65 else 1)
                    rec = "BUY" if abc >= 2 else "HOLD / TRIM"
                else:
                    bear_votes = 1 + (1 if vtrend_last == -1 else 0)
                    rec = f"SELL (XYZ-{bear_votes})"

                # --- 6. STOP LOSS (20 EMA Monthly / 13 EMA Weekly or Daily) ---
                if "Monthly" in sl_mode:
                    sl_val = float(df_m["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                elif "Weekly" in sl_mode:
                    sl_val = float(df_w["Close"].ewm(span=13, adjust=False).mean().iloc[-1])
                else:
                    sl_val = float(df_d["Close"].ewm(span=13, adjust=False).mean().iloc[-1])

                sl_buffer_pct = ((curr_close - sl_val) / curr_close) * 100.0

                results.append(
                    {
                        "Ticker": t,
                        "Last Price": round(curr_close, 2),
                        "Classification": classification,
                        "Weekly Stage": stage,
                        "DC (20)": dc_col,
                        "52W High": w52_col,
                        "BB(50,2)": bb_col,
                        "Signal": rec,
                        "Stop Loss": round(sl_val, 2),
                        "SL Buffer %": round(sl_buffer_pct, 2),
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(results)

# ==========================================
# 3. EXECUTE & RENDER STYLED TABLE
# ==========================================
if st.button("🚀 Run Multi-Stock Classification", type="primary"):
    if not tickers:
        st.warning("Please enter or upload at least one stock ticker.")
    else:
        with st.spinner(f"Scanning {len(tickers)} stocks across Monthly & Weekly timeframes..."):
            df_out = fetch_and_analyze_batch(tuple(tickers), timeframe_sl, vol_surge_mult)

        if not df_out.empty:
            # Filter controls
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                class_filter = st.multiselect(
                    "Filter by Classification",
                    options=["CORE+SATELLITE", "CORE", "SATELLITE", "Neither"],
                    default=["CORE+SATELLITE", "CORE", "SATELLITE", "Neither"],
                )
            with f_col2:
                only_breakouts = st.checkbox("Show only stocks with active breakouts (DC / 52W / BB)", value=False)

            filtered_df = df_out[df_out["Classification"].isin(class_filter)].copy()
            if only_breakouts:
                filtered_df = filtered_df[
                    (filtered_df["DC (20)"] != "-")
                    | (filtered_df["52W High"] != "-")
                    | (filtered_df["BB(50,2)"] != "-")
                ]

            # Exact color styling matching your TradingView dashboard
            def style_class(val):
                palette = {
                    "CORE+SATELLITE": "background-color: #008755; color: white; font-weight: bold",
                    "CORE": "background-color: #147341; color: white; font-weight: bold",
                    "SATELLITE": "background-color: #235aa0; color: white; font-weight: bold",
                    "Neither": "background-color: #323232; color: #bbbbbb",
                }
                return palette.get(val, "")

            def style_stage(val):
                if "Stage 2" in val:
                    return "background-color: #00aa50; color: white; font-weight: bold"
                if "Stage 1" in val:
                    return "background-color: #808080; color: white"
                if "Stage 3" in val:
                    return "background-color: #dcb400; color: black; font-weight: bold"
                if "Stage 4" in val:
                    return "background-color: #dc5064; color: white"
                return ""

            def style_dc(val):
                return "background-color: #196e2d; color: #ffbf00; font-weight: bold" if val == "DC" else "color: #666666"

            def style_52w(val):
                return "background-color: #196e2d; color: #64b4ff; font-weight: bold" if val == "52W" else "color: #666666"

            def style_bb(val):
                return "background-color: #196e2d; color: #d782ff; font-weight: bold" if val == "BB(50,2)" else "color: #666666"

            def style_signal(val):
                if "BUY" in val:
                    return "background-color: #228b22; color: white; font-weight: bold"
                if "SELL" in val:
                    return "background-color: #dc143c; color: white; font-weight: bold"
                return "background-color: #1e90ff; color: white; font-weight: bold"

            styled = (
                filtered_df.style.map(style_class, subset=["Classification"])
                .map(style_stage, subset=["Weekly Stage"])
                .map(style_dc, subset=["DC (20)"])
                .map(style_52w, subset=["52W High"])
                .map(style_bb, subset=["BB(50,2)"])
                .map(style_signal, subset=["Signal"])
                .format({"Last Price": "{:.2f}", "Stop Loss": "{:.2f}", "SL Buffer %": "{:.2f}%"})
            )

            st.dataframe(styled, use_container_width=True, height=550)

            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Results to CSV",
                data=csv_bytes,
                file_name="shaktimaan_classification_output.csv",
                mime="text/csv",
            )
        else:
            st.error("No valid stock data found. Ensure Indian stocks end with .NS (e.g., TCS.NS) and UK stocks end with .L (e.g., AZN.L).")