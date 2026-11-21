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
ADMIN_PASSWORD = "AdminSecretPin"
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

def parse_uploaded_tokens(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    return [t.strip().upper() for t in content.replace("\n", ",").split(",") if t.strip()]

if "authenticated_admin" not in st.session_state:
    st.session_state.authenticated_admin = False

if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_persisted_watchlists()

# Sidebar Setup
st.sidebar.header("📁 Watchlist Manager")
if not st.session_state.authenticated_admin:
    admin_pass_input = st.sidebar.text_input("Admin Password (to edit/delete)", type="password")
    if st.sidebar.button("Unlock Admin Mode"):
        if admin_pass_input == ADMIN_PASSWORD:
            st.session_state.authenticated_admin = True
            st.success("Admin mode unlocked!")
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")
    st.sidebar.info("Public Mode: View existing watchlists.")
else:
    st.sidebar.success("👑 Admin Mode Active")
    if st.sidebar.button("Lock Admin Mode"):
        st.session_state.authenticated_admin = False
        st.rerun()

st.sidebar.divider()
auto_append_ns = st.sidebar.checkbox("Auto-append .NS if no suffix", value=True)
vol_surge_mult = st.sidebar.number_input("Volume Surge Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

available_mls = list(st.session_state.watchlists.keys())
if st.session_state.authenticated_admin:
    available_mls.append("+ Create New Watchlist")

selected_wl = st.sidebar.selectbox("Select Stored Watchlist", available_mls)

if selected_wl == "+ Create New Watchlist" and st.session_state.authenticated_admin:
    new_wl_name_input = st.sidebar.text_input("New Watchlist Name")
    new_wl_file = st.sidebar.file_uploader("Upload Tickers (CSV / TXT)", type=["csv", "txt"], key="admin_new_wl_file")

    default_new_tickers = "TICKER1, TICKER2"
    suggested_name = new_wl_name_input.strip()
    if new_wl_file is not None:
        try:
            file_tokens = parse_uploaded_tokens(new_wl_file)
            if file_tokens:
                default_new_tickers = ", ".join(file_tokens)
                if not suggested_name:
                    suggested_name = new_wl_file.name.rsplit(".", 1)[0].replace("_", " ").title()
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

    new_wl_tickers = st.sidebar.text_area("Tickers (comma separated)", default_new_tickers, height=100)
    if st.sidebar.button("Save New Watchlist"):
        final_wl_name = new_wl_name_input.strip() or suggested_name
        if final_wl_name:
            st.session_state.watchlists[final_wl_name] = new_wl_tickers
            save_persisted_watchlists(st.session_state.watchlists)
            st.success(f"Watchlist '{final_wl_name}' saved!")
            st.rerun()
        else:
            st.sidebar.warning("Please enter a watchlist name.")
    base_tickers_text = new_wl_tickers
else:
    base_tickers_text = st.session_state.watchlists.get(selected_wl, "RELIANCE, TCS")
    if st.session_state.authenticated_admin:
        edit_upload_file = st.sidebar.file_uploader(
            f"Import CSV/TXT into '{selected_wl}'",
            type=["csv", "txt"],
            key=f"admin_edit_wl_file_{selected_wl}"
        )
        upload_action = st.sidebar.radio(
            "File Import Mode",
            ["Replace Watchlist", "Append to Watchlist"],
            horizontal=True,
            key=f"admin_edit_wl_mode_{selected_wl}"
        )
        if edit_upload_file is not None and st.sidebar.button(f"Apply File to '{selected_wl}'", key=f"apply_file_{selected_wl}"):
            try:
                file_tokens = parse_uploaded_tokens(edit_upload_file)
                if file_tokens:
                    if upload_action == "Append to Watchlist":
                        existing_tokens = [t.strip().upper() for t in base_tickers_text.replace("\n", ",").split(",") if t.strip()]
                        merged_text = ", ".join(existing_tokens + file_tokens)
                    else:
                        merged_text = ", ".join(file_tokens)
                    st.session_state.watchlists[selected_wl] = merged_text
                    save_persisted_watchlists(st.session_state.watchlists)
                    st.sidebar.success(f"Updated '{selected_wl}' from {edit_upload_file.name}!")
                    st.rerun()
                else:
                    st.sidebar.warning("Uploaded file had no valid tickers.")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

        updated_text = st.sidebar.text_area(f"Edit Tickers ({selected_wl})", base_tickers_text, height=100)
        if updated_text != base_tickers_text:
            st.session_state.watchlists[selected_wl] = updated_text
            save_persisted_watchlists(st.session_state.watchlists)
            base_tickers_text = updated_text

# ==========================================
# 1. TECHNICAL ENGINE CODE
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
            tickers=list(chunk),
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
                    "Weekly Stage": "Insufficient Data", "DC Weekly": "-", "52W High": "-", "BB(50,2)": "-",
                    "Satellite Signal": "Insufficient Data", "SL Satellite": 0.0, "SL Sat Buffer %": 0.0,
                    "DC Monthly": "-", "Core Signal": "Insufficient Data", "SL Core": 0.0, "SL Core Buffer %": 0.0,
                    "_w_red": False, "_m_red": False
                })
            continue

        for t in chunk:
            try:
                # Robust extraction supporting both single-stock (flat) and multi-stock (MultiIndex) outputs
                if len(list(set(chunk))) == 1 and len(chunk) == 1:
                    df_d = raw_data.copy()
                    if isinstance(df_d.columns, pd.MultiIndex):
                        df_d.columns = df_d.columns.droplevel(0)
                else:
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        if t not in raw_data.columns.get_level_values(0):
                            results.append({
                                "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                                "Weekly Stage": "Insufficient Data", "DC Weekly": "-", "52W High": "-", "BB(50,2)": "-",
                                "Satellite Signal": "Insufficient Data", "SL Satellite": 0.0, "SL Sat Buffer %": 0.0,
                                "DC Monthly": "-", "Core Signal": "Insufficient Data", "SL Core": 0.0, "SL Core Buffer %": 0.0,
                                "_w_red": False, "_m_red": False
                            })
                            continue
                        df_d = raw_data[t].dropna(how="all").copy()
                    else:
                        df_d = raw_data.copy()

                df_d = df_d.dropna(subset=["Close", "High", "Low", "Volume"])
                if len(df_d) < 50:
                    results.append({
                        "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                        "Weekly Stage": "Insufficient Data", "DC Weekly": "-", "52W High": "-", "BB(50,2)": "-",
                        "Satellite Signal": "Insufficient Data", "SL Satellite": 0.0, "SL Sat Buffer %": 0.0,
                        "DC Monthly": "-", "Core Signal": "Insufficient Data", "SL Core": 0.0, "SL Core Buffer %": 0.0,
                        "_w_red": False, "_m_red": False
                    })
                    continue

                agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                df_w = df_d.resample("W-FRI").agg(agg_rules).dropna()
                df_m = df_d.resample("ME").agg(agg_rules).dropna()

                curr_close = float(df_d["Close"].iloc[-1])
                clean_ticker = t.split(".")[0]
                exchange_prefix = "LMS" if t.endswith(".L") else ("BSE" if t.endswith(".BO") else "NSE")
                tv_link = f"https://in.tradingview.com/chart/GCzoFDBK/?symbol={exchange_prefix}%3A{clean_ticker}"

                sl_sat = float(df_w["Close"].ewm(span=13, adjust=False).mean().iloc[-1]) if len(df_w) >= 13 else curr_close * 0.95
                sl_sat_buffer = ((curr_close - sl_sat) / curr_close) * 100.0

                m_open = float(df_m["Open"].iloc[-1]) if not df_m.empty else curr_close
                m_ema20 = float(df_m["Close"].ewm(span=20, adjust=False).mean().iloc[-1]) if len(df_m) >= 20 else curr_close * 0.95
                max_allowable_core_sl = m_open * 0.80
                sl_core = max(m_ema20, max_allowable_core_sl)
                sl_core_buffer = ((curr_close - sl_core) / curr_close) * 100.0

                if len(df_w) >= 6:
                    w_ema20_s = df_w["Close"].ewm(span=20, adjust=False).mean()
                    w_ema20 = float(w_ema20_s.iloc[-1])
                    w_ema20_past = float(w_ema20_s.iloc[-6])
                    w_slope = (w_ema20 - w_ema20_past) / w_ema20_past
                    w_close_val = float(df_w["Close"].iloc[-1])
                    if w_close_val > w_ema20:
                        stage = "Stage 1: Basing" if w_slope < -0.0005 else "Stage 2: Advancing"
                    else:
                        stage = "Stage 3: Distribution" if w_slope >= -0.0005 else "Stage 4: Declining"
                else:
                    stage = "Insufficient Data"

                w_close_val = float(df_w["Close"].iloc[-1]) if not df_w.empty else 0
                w_open_val = float(df_w["Open"].iloc[-1]) if not df_w.empty else 0
                w_red = w_close_val < w_open_val

                m_close_val = float(df_m["Close"].iloc[-1]) if not df_m.empty else 0
                m_open_val = float(df_m["Open"].iloc[-1]) if not df_m.empty else 0
                m_red = m_close_val < m_open_val

                ref_high = float(df_d["High"].iloc[-1])
                is_52w = ref_high >= float(df_d["High"].iloc[-253:-1].max()) if len(df_d) >= 253 else False

                is_dc_w = False
                if len(df_w) >= 21:
                    w_high_curr = float(df_w["High"].iloc[-1])
                    w_high_prev_max = float(df_w["High"].iloc[-21:-1].max())
                    is_dc_w = w_high_curr > w_high_prev_max

                is_dc_m = False
                if len(df_m) >= 21:
                    m_high_curr = float(df_m["High"].iloc[-1])
                    m_high_prev_max = float(df_m["High"].iloc[-21:-1].max())
                    is_dc_m = m_high_curr > m_high_prev_max

                if len(df_w) >= 50:
                    w_close_series = df_w["Close"]
                    bb_basis = float(w_close_series.rolling(50).mean().iloc[-1])
                    bb_std = float(w_close_series.rolling(50).std(ddof=1).iloc[-1])
                    upper_band = bb_basis + (2.0 * bb_std)
                    is_bb = w_close_val > upper_band
                else:
                    is_bb = False

                dc_w_col = "DC" if is_dc_w else "-"
                dc_m_col = "DC" if is_dc_m else "-"
                w52_col = "52W" if is_52w else "-"
                bb_col = "BB(50,2)" if is_bb else "-"

                if len(df_w) >= 14:
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
                else:
                    sat_signal = "Insufficient Data"

                if len(df_m) >= 14:
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
                else:
                    core_signal = "Insufficient Data"

                if is_dc_w and sat_signal == "BUY" and w_close_val > w_open_val:
                    sat_signal = "BUY / ADD"

                if is_dc_m and core_signal == "BUY" and m_close_val > m_open_val:
                    core_signal = "BUY / ADD"

                results.append({
                    "Ticker": t, "Chart": tv_link, "Clean Ticker": clean_ticker, "Last Price": round(curr_close, 2),
                    "Weekly Stage": stage, "DC Weekly": dc_w_col, "52W High": w52_col, "BB(50,2)": bb_col,
                    "Satellite Signal": sat_signal, "SL Satellite": round(sl_sat, 2), "SL Sat Buffer %": round(sl_sat_buffer, 2),
                    "DC Monthly": dc_m_col, "Core Signal": core_signal, "SL Core": round(sl_core, 2), "SL Core Buffer %": round(sl_core_buffer, 2),
                    "_w_red": w_red, "_m_red": m_red
                })
            except Exception as e:
                results.append({
                    "Ticker": t, "Chart": "", "Clean Ticker": t.split(".")[0], "Last Price": 0.0,
                    "Weekly Stage": "Insufficient Data", "DC Weekly": "-", "52W High": "-", "BB(50,2)": "-",
                    "Satellite Signal": "Insufficient Data", "SL Satellite": 0.0, "SL Sat Buffer %": 0.0,
                    "DC Monthly": "-", "Core Signal": "Insufficient Data", "SL Core": 0.0, "SL Core Buffer %": 0.0,
                    "_w_red": False, "_m_red": False
                })

    return pd.DataFrame(results)

# ==========================================
# 2. MAIN DASHBOARD VIEW
# ==========================================
st.title("📊 Shaktimaan Hybrid — Multi-Stock Stop Loss & Signal Dashboard")

# --- PROMINENT MAIN PAGE INPUT SECTION ---
st.markdown("### ✏️ Scan Input")
input_mode = st.radio("Choose Input Method", ["Use Selected Watchlist Stored Data", "Type / Paste Custom Tickers"], horizontal=True)

if input_mode == "Type / Paste Custom Tickers":
    user_manual_input = st.text_input("Enter Tickers (comma separated)", "RELIANCE, TCS, INFY, ATHERENERG, PPLPHARMA, HDFCLIFE")
    active_tokens_text = user_manual_input
else:
    active_tokens_text = base_tickers_text
    st.info(f"Active Stored Watchlist: **{selected_wl}** (`{base_tickers_text}`)")

parsed_tokens = [t.strip().upper() for t in active_tokens_text.replace("\n", ",").split(",") if t.strip()]
tickers = []
for t in parsed_tokens:
    if auto_append_ns and "." not in t:
        tickers.append(f"{t}.NS")
    else:
        tickers.append(t)

st.write(f"Tokens queued for scan: `{', '.join(tickers)}`")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if st.button("🚀 Run Classification & Stop Loss Scan", type="primary"):
    if not tickers:
        st.warning("Selected watchlist or manual input is empty.")
    else:
        with st.spinner(f"Scanning {len(tickers)} stocks..."):
            st.session_state.scan_results = fetch_and_analyze_batch(tuple(tickers), vol_surge_mult)

# Render results and filters if scan data exists in session state
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df_out = st.session_state.scan_results

    st.markdown("---")
    st.markdown("### 🔍 Filter Results")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        stage_options = ["Stage 1: Basing", "Stage 2: Advancing", "Stage 3: Distribution", "Stage 4: Declining", "Insufficient Data"]
        selected_stages = st.multiselect("Weekly Stage", options=stage_options, default=stage_options, key="filter_stage")
    with f_col2:
        search_query = st.text_input("Ticker Search", "", key="filter_search").strip().upper()
    with f_col3:
        signal_options = ["BUY", "BUY / ADD", "HOLD / TRIM", "SELL", "Insufficient Data"]
        selected_signal = st.selectbox("Signal Filter (Satellite)", options=["All"] + signal_options, key="filter_signal")
    with f_col4:
        only_breakouts = st.checkbox("Active Breakouts Only", value=False, key="filter_breakouts")

    filtered_df = df_out.copy()
    if selected_stages:
        filtered_df = filtered_df[filtered_df["Weekly Stage"].apply(lambda x: any(s in x for s in selected_stages))]
    if search_query:
        filtered_df = filtered_df[filtered_df["Ticker"].str.contains(search_query)]
    if selected_signal != "All":
        filtered_df = filtered_df[filtered_df["Satellite Signal"].str.contains(selected_signal)]
    if only_breakouts:
        filtered_df = filtered_df[
            (filtered_df["DC Weekly"] != "-")
            | (filtered_df["DC Monthly"] != "-")
            | (filtered_df["52W High"] != "-")
            | (filtered_df["BB(50,2)"] != "-")
        ]

    if filtered_df.empty:
        st.warning("No stocks match the selected filter criteria. Try expanding your filter options.")
    else:
        def highlight_cells(row):
            styles = [''] * len(row)
            col_map = {col: i for i, col in enumerate(row.index)}

            stage = row.get("Weekly Stage", "")
            if "Stage 2" in stage:
                styles[col_map["Weekly Stage"]] = "background-color: #00aa50; color: white; font-weight: bold"
            elif "Stage 1" in stage:
                styles[col_map["Weekly Stage"]] = "background-color: #808080; color: white"
            elif "Stage 3" in stage:
                styles[col_map["Weekly Stage"]] = "background-color: #dcb400; color: black; font-weight: bold"
            elif "Stage 4" in stage:
                styles[col_map["Weekly Stage"]] = "background-color: #dc5064; color: white"
            elif "Insufficient Data" in stage:
                styles[col_map["Weekly Stage"]] = "background-color: #555555; color: #ffaaaa; font-style: italic"

            dc_w = row.get("DC Weekly", "-")
            w_red = row.get("_w_red", False)
            if "DC" in str(dc_w):
                styles[col_map["DC Weekly"]] = f"background-color: #196e2d; color: {'#ffbf00' if w_red else '#ffffff'}; font-weight: bold"
            else:
                styles[col_map["DC Weekly"]] = "color: #666666"

            dc_m = row.get("DC Monthly", "-")
            m_red = row.get("_m_red", False)
            if "DC" in str(dc_m):
                styles[col_map["DC Monthly"]] = f"background-color: #196e2d; color: {'#ffbf00' if m_red else '#ffffff'}; font-weight: bold"
            else:
                styles[col_map["DC Monthly"]] = "color: #666666"

            if row.get("52W High", "-") == "52W":
                styles[col_map["52W High"]] = "background-color: #196e2d; color: #64b4ff; font-weight: bold"
            else:
                styles[col_map["52W High"]] = "color: #666666"

            if row.get("BB(50,2)", "-") == "BB(50,2)":
                styles[col_map["BB(50,2)"]] = "background-color: #196e2d; color: #d782ff; font-weight: bold"
            else:
                styles[col_map["BB(50,2)"]] = "color: #666666"

            for sig_col in ["Satellite Signal", "Core Signal"]:
                if sig_col in col_map:
                    sig = row.get(sig_col, "")
                    if "BUY" in sig:
                        styles[col_map[sig_col]] = "background-color: #228b22; color: white; font-weight: bold"
                    elif "SELL" in sig:
                        styles[col_map[sig_col]] = "background-color: #dc143c; color: white; font-weight: bold"
                    elif "HOLD" in sig:
                        styles[col_map[sig_col]] = "background-color: #1e90ff; color: white; font-weight: bold"
                    else:
                        styles[col_map[sig_col]] = "background-color: #555555; color: #ffaaaa; font-style: italic"

            return styles

        display_columns = [
            "Chart", "Ticker", "Last Price", "Weekly Stage", 
            "DC Weekly", "52W High", "BB(50,2)", "Satellite Signal", "SL Satellite", "SL Sat Buffer %", 
            "DC Monthly", "Core Signal", "SL Core", "SL Core Buffer %"
        ]

        styled_df = (
            filtered_df[display_columns + ["_w_red", "_m_red"]]
            .style
            .apply(highlight_cells, axis=1)
            .hide(subset=["_w_red", "_m_red"], axis="columns")
            .format({
                "Last Price": "{:.2f}", 
                "SL Satellite": "{:.2f}", 
                "SL Sat Buffer %": "{:.2f}%",
                "SL Core": "{:.2f}",
                "SL Core Buffer %": "{:.2f}%"
            })
        )

        st.dataframe(
            styled_df, 
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

        csv_bytes = filtered_df.drop(columns=["Chart", "Clean Ticker", "_w_red", "_m_red"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Results to CSV",
            data=csv_bytes,
            file_name="shaktimaan_scan_results.csv",
            mime="text/csv",
        )
elif st.session_state.scan_results is not None and st.session_state.scan_results.empty:
    st.error("No valid stock data returned.")
