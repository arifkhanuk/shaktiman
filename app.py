import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Shaktimaan Hybrid Scanner", layout="wide")

# ==========================================
# 0. PERSISTENT WATCHLISTS & ADMIN SETUP
# ==========================================
ADMIN_PASSWORD = "AdminSecretPin"  # Password required to edit/add/delete watchlists
WATCHLIST_FILE = "watchlists.json"

def load_persisted_watchlists():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "Core Portfolio": "RELIANCE, TCS, HDFCBANK, INFY, BHARTIARTL",
        "Breakout Watchlist": "ICICIBANK, SBIN, LT, COFORGE, ZYUSLIFE",
        "US Tech & Global": "AAPL, MSFT, NVDA, AZN, SHEL"
    }

def save_persisted_watchlists(wl_dict):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(wl_dict, f, indent=4)
    except Exception:
        pass

if "authenticated_admin" not in st.session_state:
    st.session_state.authenticated_admin = False

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_persisted_watchlists()

st.sidebar.header("📁 Watchlist Manager")

# Admin Control Panel in Sidebar
if not st.session_state.authenticated_admin:
    admin_pass_input = st.sidebar.text_input("Admin Password (to edit/delete)", type="password")
    if st.sidebar.button("Unlock Admin Mode"):
        if admin_pass_input == ADMIN_PASSWORD:
            st.session_state.authenticated_admin = True
            st.success("Admin mode unlocked!")
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")
    st.sidebar.info("Public Mode: View existing watchlists & upload session files.")
else:
    st.sidebar.success("👑 Admin Mode Active")
    if st.sidebar.button("Lock Admin Mode"):
        st.session_state.authenticated_admin = False
        st.rerun()

st.sidebar.divider()

# Public file uploader (Session-only storage for public users, persistent if admin)
uploaded_file = st.sidebar.file_uploader("Upload Temporary Watchlist (CSV / TXT)", type=["csv", "txt"])
session_custom_watchlist = None
if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8")
        file_tokens = [t.strip().upper() for t in content.replace("\n", ",").split(",") if t.strip()]
        if file_tokens:
            wl_name = "Uploaded: " + uploaded_file.name.split(".")[0].replace("_", " ").title()
            if st.session_state.authenticated_admin:
                st.session_state.watchlists[wl_name] = ", ".join(file_tokens)
                save_persisted_watchlists(st.session_state.watchlists)
            session_custom_watchlist = (wl_name, ", ".join(file_tokens))
            st.sidebar.success(f"Loaded '{wl_name}'!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Build selection list (including manual custom ticker input option)
available_mls = list(st.session_state.watchlists.keys())
if session_custom_watchlist and session_custom_watchlist[0] not in available_mls:
    available_mls.insert(0, session_custom_watchlist[0])
available_mls.append("✏️ Enter Manual Tickers")
if st.session_state.authenticated_admin:
    available_mls.append("+ Create New Watchlist")

selected_wl = st.sidebar.selectbox("Select Active Watchlist", available_mls)

# Handle Watchlist Content / Manual Input
if selected_wl == "✏️ Enter Manual Tickers":
    active_tickers_text = st.sidebar.text_area("Type Tickers (comma or newline separated)", "RELIANCE, TCS, INFY, ATHERENERG, PPLPHARMA", height=120, help="Type or paste your stock symbols directly here.")
    st.sidebar.caption("⚠️ Custom manual input for this session.")
elif selected_wl == "+ Create New Watchlist" and st.session_state.authenticated_admin:
    new_wl_name = st.sidebar.text_input("New Watchlist Name")
    new_wl_tickers = st.sidebar.text_area("Tickers (comma separated)", "TICKER1, TICKER2")
    if st.sidebar.button("Save New Watchlist"):
        if new_wl_name.strip():
            st.session_state.watchlists[new_wl_name.strip()] = new_wl_tickers
            save_persisted_watchlists(st.session_state.watchlists)
            st.success(f"Watchlist '{new_wl_name}' saved persistently!")
            st.rerun()
    active_tickers_text = "TICKER1, TICKER2"
elif session_custom_watchlist and selected_wl == session_custom_watchlist[0]:
    active_tickers_text = session_custom_watchlist[1]
    st.sidebar.caption("⚠️ Temporary session watchlist.")
else:
    active_tickers_text = st.session_state.watchlists.get(selected_wl, "RELIANCE, TCS")
    if st.session_state.authenticated_admin:
        updated_text = st.sidebar.text_area(f"Edit Tickers ({selected_wl})", active_tickers_text, height=120)
        if updated_text != active_tickers_text:
            st.session_state.watchlists[selected_wl] = updated_text
            save_persisted_watchlists(st.session_state.watchlists)
            active_tickers_text = updated_text

        if st.sidebar.button(f"🗑️ Delete '{selected_wl}'"):
            if len(st.session_state.watchlists) > 1:
                del st.session_state.watchlists[selected_wl]
                save_persisted_watchlists(st.session_state.watchlists)
                st.sidebar.success("Deleted successfully!")
                st.rerun()
            else:
                st.sidebar.warning("Must keep at least one permanent watchlist.")

st.sidebar.divider()
auto_append_ns = st.sidebar.checkbox("Auto-append .NS if no suffix", value=True)
vol_surge_mult = st.sidebar.number_input("Volume Surge Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Parse tickers (handling both commas and newlines)
parsed_tokens = [t.strip().upper() for t in active_tickers_text.replace("\n", ",").split(",") if t.strip()]
tickers = []
for t in parsed_tokens:
    if auto_append_ns and "." not in t:
        tickers.append(f"{t}.NS")
    else:
        tickers.append(t)
tickers = list(dict.fromkeys(tickers))

# ==========================================
# 1. TECHNICAL ENGINE
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
def fetch_and_analyze_batch(ticker_tuple: tuple, vol_mult: float) -> pd.DataFrame:
    ticker_list = list(ticker_tuple)
    results = []
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
            for t in chunk:
                results.append({
                    "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                    "Weekly Stage": "Insufficient Data", "DC (20)": "-", "52W High": "-", "BB(50,2)": "-",
                    "Satellite Signal": "Insufficient Data", "Core Signal": "Insufficient Data",
                    "SL Satellite": 0.0, "SL Sat Buffer %": 0.0, "SL Core": 0.0, "SL Core Buffer %": 0.0,
                    "Classification": "Insufficient Data"
                })
            continue

        for t in chunk:
            try:
                if len(chunk) == 1:
                    df_d = raw_data.copy()
                    if isinstance(df_d.columns, pd.MultiIndex):
                        df_d.columns = df_d.columns.get_level_values(0)
                else:
                    if t not in raw_data.columns.get_level_values(0):
                        results.append({
                            "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                            "Weekly Stage": "Insufficient Data", "DC (20)": "-", "52W High": "-", "BB(50,2)": "-",
                            "Satellite Signal": "Insufficient Data", "Core Signal": "Insufficient Data",
                            "SL Satellite": 0.0, "SL Sat Buffer %": 0.0, "SL Core": 0.0, "SL Core Buffer %": 0.0,
                            "Classification": "Insufficient Data"
                        })
                        continue
                    df_d = raw_data[t].dropna(how="all").copy()

                df_d = df_d.dropna(subset=["Close", "High", "Low", "Volume"])
                if len(df_d) < 252:
                    results.append({
                        "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                        "Weekly Stage": "Insufficient Data", "DC (20)": "-", "52W High": "-", "BB(50,2)": "-",
                        "Satellite Signal": "Insufficient Data", "Core Signal": "Insufficient Data",
                        "SL Satellite": 0.0, "SL Sat Buffer %": 0.0, "SL Core": 0.0, "SL Core Buffer %": 0.0,
                        "Classification": "Insufficient Data"
                    })
                    continue

                agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                df_w = df_d.resample("W-FRI").agg(agg_rules).dropna()
                df_m = df_d.resample("ME").agg(agg_rules).dropna()

                if len(df_m) < 61 or len(df_w) < 53:
                    results.append({
                        "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": float(df_d["Close"].iloc[-1]) if not df_d.empty else 0.0,
                        "Weekly Stage": "Insufficient Data", "DC (20)": "-", "52W High": "-", "BB(50,2)": "-",
                        "Satellite Signal": "Insufficient Data", "Core Signal": "Insufficient Data",
                        "SL Satellite": 0.0, "SL Sat Buffer %": 0.0, "SL Core": 0.0, "SL Core Buffer %": 0.0,
                        "Classification": "Insufficient Data"
                    })
                    continue

                curr_close = float(df_d["Close"].iloc[-1])

                # 1. Monthly Core
                m_close = float(df_m["Close"].iloc[-1])
                m_sma20 = float(df_m["Close"].rolling(20).mean().iloc[-1])
                m_sma60 = float(df_m["Close"].rolling(60).mean().iloc[-1])
                m_close12 = float(df_m["Close"].iloc[-13])
                m_roc12 = ((m_close - m_close12) / m_close12) * 100.0
                tf_core = (m_close > m_sma60) and (m_sma20 > m_sma60) and (m_roc12 > 15.0)

                # 2. Weekly Satellite
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

                # 3. Weekly Stage
                w_ema20_s = df_w["Close"].ewm(span=20, adjust=False).mean()
                w_ema20 = float(w_ema20_s.iloc[-1])
                w_ema20_past = float(w_ema20_s.iloc[-6])
                w_slope = (w_ema20 - w_ema20_past) / w_ema20_past

                if w_close > w_ema20:
                    stage = "Stage 1: Basing" if w_slope < -0.0005 else "Stage 2: Advancing"
                else:
                    stage = "Stage 3: Distribution" if w_slope >= -0.0005 else "Stage 4: Declining"

                # 4. Breakout Triggers
                ref_high = float(df_d["High"].iloc[-1])
                ref_close = float(df_d["Close"].iloc[-1])
                is_52w = ref_high >= float(df_d["High"].iloc[-253:-1].max())
                is_dc = ref_high > float(df_d["High"].iloc[-21:-1].max())

                bb_basis = df_d["Close"].rolling(50).mean().iloc[-1]
                bb_std = df_d["Close"].rolling(50).std().iloc[-1]
                is_bb = ref_close > (bb_basis + 2.0 * bb_std)

                dc_col = "DC" if is_dc else "-"
                w52_col = "52W" if is_52w else "-"
                bb_col = "BB(50,2)" if is_bb else "-"

                # 5. Satellite Signal (Weekly Candles df_w)
                rsi_w = compute_rsi(df_w["Close"], 14)
                rsi_w_ma = rsi_w.rolling(14).mean()
                curr_rsi_w = float(rsi_w.iloc[-1])
                curr_rsi_w_ma = float(rsi_w_ma.iloc[-1])
                _, vtrend_w_last = compute_vstop(df_w["Close"], df_w["High"], df_w["Low"])

                if curr_rsi_w >= 55:
                    if curr_rsi_w >= curr_rsi_w_ma:
                        abc_w = 3 if curr_rsi_w > 70 else (2 if curr_rsi_w >= 60 else 1)
                    else:
                        abc_w = 3 if curr_rsi_w > 74 else (2 if curr_rsi_w >= 65 else 1)
                    sat_signal = "BUY" if abc_w >= 2 else "HOLD / TRIM"
                else:
                    bear_votes_w = 1 + (1 if vtrend_w_last == -1 else 0)
                    sat_signal = f"SELL (XYZ-{bear_votes_w})"

                # 6. Core Signal (Monthly Candles df_m)
                rsi_m = compute_rsi(df_m["Close"], 14)
                rsi_m_ma = rsi_m.rolling(14).mean()
                curr_rsi_m = float(rsi_m.iloc[-1])
                curr_rsi_m_ma = float(rsi_m_ma.iloc[-1])
                _, vtrend_m_last = compute_vstop(df_m["Close"], df_m["High"], df_m["Low"])

                if curr_rsi_m >= 55:
                    if curr_rsi_m >= curr_rsi_m_ma:
                        abc_m = 3 if curr_rsi_m > 70 else (2 if curr_rsi_m >= 60 else 1)
                    else:
                        abc_m = 3 if curr_rsi_m > 74 else (2 if curr_rsi_m >= 65 else 1)
                    core_signal = "BUY" if abc_m >= 2 else "HOLD / TRIM"
                else:
                    bear_votes_m = 1 + (1 if vtrend_m_last == -1 else 0)
                    core_signal = f"SELL (XYZ-{bear_votes_m})"

                # 7. Dual Stop Losses & Buffers
                sl_sat = float(df_w["Close"].ewm(span=13, adjust=False).mean().iloc[-1])
                sl_sat_buffer = ((curr_close - sl_sat) / curr_close) * 100.0

                m_open = float(df_m["Open"].iloc[-1])
                m_ema20 = float(df_m["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                max_allowable_core_sl = m_open * 0.80
                sl_core = max(m_ema20, max_allowable_core_sl)
                sl_core_buffer = ((curr_close - sl_core) / curr_close) * 100.0

                clean_ticker = t.split(".")[0]
                exchange_prefix = "LMS" if t.endswith(".L") else ("BSE" if t.endswith(".BO") else "NSE")
                tv_link = f"https://in.tradingview.com/chart/GCzoFDBK/?symbol={exchange_prefix}%3A{clean_ticker}"

                results.append(
                    {
                        "Ticker": t,
                        "Chart": tv_link,
                        "Clean Ticker": clean_ticker,
                        "Last Price": round(curr_close, 2),
                        "Weekly Stage": stage,
                        "DC (20)": dc_col,
                        "52W High": w52_col,
                        "BB(50,2)": bb_col,
                        "Satellite Signal": sat_signal,
                        "Core Signal": core_signal,
                        "SL Satellite": round(sl_sat, 2),
                        "SL Sat Buffer %": round(sl_sat_buffer, 2),
                        "SL Core": round(sl_core, 2),
                        "SL Core Buffer %": round(sl_core_buffer, 2),
                        "Classification": classification
                    }
                )
            except Exception:
                results.append({
                    "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                    "Weekly Stage": "Insufficient Data", "DC (20)": "-", "52W High": "-", "BB(50,2)": "-",
                    "Satellite Signal": "Insufficient Data", "Core Signal": "Insufficient Data",
                    "SL Satellite": 0.0, "SL Sat Buffer %": 0.0, "SL Core": 0.0, "SL Core Buffer %": 0.0,
                    "Classification": "Insufficient Data"
                })

    return pd.DataFrame(results)

# ==========================================
# 2. MAIN DASHBOARD VIEW
# ==========================================
st.title("📊 Shaktimaan Hybrid — Multi-Stock Classification & Stop Loss Dashboard")
st.subheader(f"Active Watchlist: `{selected_wl}`")
st.write(f"Tokens to scan: `{', '.join(tickers)}`")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if st.button("🚀 Run Classification & Stop Loss Scan", type="primary"):
    if not tickers:
        st.warning("Selected watchlist is empty.")
    else:
        with st.spinner(f"Scanning {len(tickers)} stocks for '{selected_wl}'..."):
            st.session_state.scan_results = fetch_and_analyze_batch(tuple(tickers), vol_surge_mult)

# Render results and filters if scan data exists in session state
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df_out = st.session_state.scan_results

    st.markdown("### 🔍 Filter Results")
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
    with f_col1:
        class_options = ["CORE+SATELLITE", "CORE", "SATELLITE", "Neither", "Insufficient Data"]
        selected_classes = st.multiselect("Classification", options=class_options, default=class_options, key="filter_class")
    with f_col2:
        stage_options = ["Stage 1: Basing", "Stage 2: Advancing", "Stage 3: Distribution", "Stage 4: Declining", "Insufficient Data"]
        selected_stages = st.multiselect("Weekly Stage", options=stage_options, default=stage_options, key="filter_stage")
    with f_col3:
        search_query = st.text_input("Ticker Search", "", key="filter_search").strip().upper()
    with f_col4:
        signal_options = ["BUY", "HOLD / TRIM", "SELL", "Insufficient Data"]
        selected_signal = st.selectbox("Signal Filter (Satellite)", options=["All"] + signal_options, key="filter_signal")
    with f_col5:
        only_breakouts = st.checkbox("Active Breakouts Only", value=False, key="filter_breakouts")

    # Apply filtering logic safely
    filtered_df = df_out.copy()
    if selected_classes:
        filtered_df = filtered_df[filtered_df["Classification"].isin(selected_classes)]
    if selected_stages:
        filtered_df = filtered_df[filtered_df["Weekly Stage"].apply(lambda x: any(s in x for s in selected_stages))]
    if search_query:
        filtered_df = filtered_df[filtered_df["Ticker"].str.contains(search_query)]
    if selected_signal != "All":
        filtered_df = filtered_df[filtered_df["Satellite Signal"].str.contains(selected_signal)]
    if only_breakouts:
        filtered_df = filtered_df[
            (filtered_df["DC (20)"] != "-")
            | (filtered_df["52W High"] != "-")
            | (filtered_df["BB(50,2)"] != "-")
        ]

    if filtered_df.empty:
        st.warning("No stocks match the selected filter criteria. Try expanding your filter options.")
    else:
        def style_class(val):
            palette = {
                "CORE+SATELLITE": "background-color: #008755; color: white; font-weight: bold",
                "CORE": "background-color: #147341; color: white; font-weight: bold",
                "SATELLITE": "background-color: #235aa0; color: white; font-weight: bold",
                "Neither": "background-color: #323232; color: #bbbbbb",
                "Insufficient Data": "background-color: #555555; color: #ffaaaa; font-style: italic"
            }
            return palette.get(val, "")

        def style_stage(val):
            if "Stage 2" in val: return "background-color: #00aa50; color: white; font-weight: bold"
            if "Stage 1" in val: return "background-color: #808080; color: white"
            if "Stage 3" in val: return "background-color: #dcb400; color: black; font-weight: bold"
            if "Stage 4" in val: return "background-color: #dc5064; color: white"
            if "Insufficient Data" in val: return "background-color: #555555; color: #ffaaaa; font-style: italic"
            return ""

        def style_dc(val):
            return "background-color: #196e2d; color: #ffbf00; font-weight: bold" if val == "DC" else "color: #666666"

        def style_52w(val):
            return "background-color: #196e2d; color: #64b4ff; font-weight: bold" if val == "52W" else "color: #666666"

        def style_bb(val):
            return "background-color: #196e2d; color: #d782ff; font-weight: bold" if val == "BB(50,2)" else "color: #666666"

        def style_signal(val):
            if "BUY" in val: return "background-color: #228b22; color: white; font-weight: bold"
            if "SELL" in val: return "background-color: #dc143c; color: white; font-weight: bold"
            if "HOLD" in val: return "background-color: #1e90ff; color: white; font-weight: bold"
            return "background-color: #555555; color: #ffdesired; font-style: italic"

        display_columns = [
            "Chart", "Ticker", "Last Price", "Weekly Stage", "DC (20)", "52W High", 
            "BB(50,2)", "Satellite Signal", "Core Signal", "SL Satellite", 
            "SL Sat Buffer %", "SL Core", "SL Core Buffer %", "Classification"
        ]

        st.dataframe(
            filtered_df[display_columns].style
            .map(style_class, subset=["Classification"])
            .map(style_stage, subset=["Weekly Stage"])
            .map(style_dc, subset=["DC (20)"])
            .map(style_52w, subset=["52W High"])
            .map(style_bb, subset=["BB(50,2)"])
            .map(style_signal, subset=["Satellite Signal", "Core Signal"])
            .format({
                "Last Price": "{:.2f}", 
                "SL Satellite": "{:.2f}", 
                "SL Sat Buffer %": "{:.2f}%",
                "SL Core": "{:.2f}",
                "SL Core Buffer %": "{:.2f}%"
            }), 
            use_container_width=True, 
            height=550,
            column_config={
                "Chart": st.column_config.LinkColumn(
                    "TradingView Chart",
                    help="Click to open chart on TradingView",
                    display_text=r"symbol=(?:.*)%3A(.*)",
                    pinned=True
                )
            }
        )

        csv_bytes = filtered_df.drop(columns=["Chart", "Clean Ticker"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Export '{selected_wl}' Results to CSV",
            data=csv_bytes,
            file_name=f"{selected_wl.lower().replace(' ', '_')}_scan.csv",
            mime="text/csv",
        )
elif st.session_state.scan_results is not None and st.session_state.scan_results.empty:
    st.error("No valid stock data returned.")
