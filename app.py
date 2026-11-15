import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Shaktimaan Hybrid Scanner", layout="wide")

# ==========================================
# 0. ACCESS CONTROL & WATCHLIST SETUP
# ==========================================
ADMIN_PASSWORD = "AdminSecretPin"  # Password required to edit/add/delete watchlists

if "authenticated_admin" not in st.session_state:
    st.session_state.authenticated_admin = False

if "watchlists" not in st.session_state:
    st.session_state.watchlists = {
        "Core Portfolio": "RELIANCE, TCS, HDFCBANK, INFY, BHARTIARTL",
        "Breakout Watchlist": "ICICIBANK, SBIN, LT, COFORGE, ZYUSLIFE",
        "US Tech & Global": "AAPL, MSFT, NVDA, AZN, SHEL"
    }

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
            session_custom_watchlist = (wl_name, ", ".join(file_tokens))
            st.sidebar.success(f"Loaded '{wl_name}'!")
    except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# Build selection list
available_mls = list(st.session_state.watchlists.keys())
if session_custom_watchlist and session_custom_watchlist[0] not in available_mls:
    available_mls.insert(0, session_custom_watchlist[0])
if st.session_state.authenticated_admin:
    available_mls.append("+ Create New Watchlist")

selected_wl = st.sidebar.selectbox("Select Active Watchlist", available_mls)

# Handle Watchlist Content
if selected_wl == "+ Create New Watchlist" and st.session_state.authenticated_admin:
    new_wl_name = st.sidebar.text_input("New Watchlist Name")
    new_wl_tickers = st.sidebar.text_area("Tickers (comma separated)", "TICKER1, TICKER2")
    if st.sidebar.button("Save New Watchlist"):
        if new_wl_name.strip():
            st.session_state.watchlists[new_wl_name.strip()] = new_wl_tickers
            st.success(f"Watchlist '{new_wl_name}' saved!")
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
            active_tickers_text = updated_text

        if st.sidebar.button(f"🗑️ Delete '{selected_wl}'"):
            if len(st.session_state.watchlists) > 1:
                del st.session_state.watchlists[selected_wl]
                st.sidebar.success("Deleted successfully!")
                st.rerun()
            else:
                st.sidebar.warning("Must keep at least one permanent watchlist.")

st.sidebar.divider()
auto_append_ns = st.sidebar.checkbox("Auto-append .NS if no suffix", value=True)
vol_surge_mult = st.sidebar.number_input("Volume Surge Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

# Parse tickers
parsed_tokens = [t.strip().upper() for t in active_tickers_text.replace("\n", ",").split(",") if t.strip()]
tickers = []
for t in parsed_tokens:
    if auto_append_ns and "." not in t:
        tickers.append(f"{t}.NS")
    else:
        tickers.append(t)
tickers = list(dict.fromkeys(tickers))

# ==========================================
# 1. TECHNICAL ENGINE (PINE SCRIPT REPLICA)
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

                agg_rules = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                df_w = df_d.resample("W-FRI").agg(agg_rules).dropna()
                df_m = df_d.resample("ME").agg(agg_rules).dropna()

                if len(df_m) < 61 or len(df_w) < 53:
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

                # 5. Signal & Action
                rsi_s = compute_rsi(df_d["Close"], 14)
                rsi_ma_s = rsi_s.rolling(14).mean()
                curr_rsi = float(rsi_s.iloc[-1])
                curr_rsi_ma = float(rsi_ma_s.iloc[-1])
                _, vtrend_last = compute_vstop(df_d["Close"], df_d["High"], df_d["Low"])

                if curr_rsi >= 55:
                    if curr_rsi >= curr_rsi_ma:
                        abc = 3 if curr_rsi > 70 else (2 if curr_rsi >= 60 else 1)
                    else:
                        abc = 3 if curr_rsi > 74 else (2 if curr_rsi >= 65 else 1)
                    rec = "BUY" if abc >= 2 else "HOLD / TRIM"
                else:
                    bear_votes = 1 + (1 if vtrend_last == -1 else 0)
                    rec = f"SELL (XYZ-{bear_votes})"

                # 6. Dual Stop Losses & Buffers
                sl_sat = float(df_w["Close"].ewm(span=13, adjust=False).mean().iloc[-1])
                sl_sat_buffer = ((curr_close - sl_sat) / curr_close) * 100.0

                m_open = float(df_m["Open"].iloc[-1])
                m_ema20 = float(df_m["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
                max_allowable_core_sl = m_open * 0.80
                sl_core = max(m_ema20, max_allowable_core_sl)
                sl_core_buffer = ((curr_close - sl_core) / curr_close) * 100.0

                # TradingView absolute URL with suffix stripped (e.g., RELIANCE.NS -> RELIANCE)
                tv_symbol = t.split(".")[0]
                tv_link = f"https://in.tradingview.com/chart/GCzoFDBK/?symbol={tv_symbol}"

                results.append(
                    {
                        "Ticker": t,
                        "Chart": tv_link,
                        "Last Price": round(curr_close, 2),
                        "Classification": classification,
                        "Weekly Stage": stage,
                        "DC (20)": dc_col,
                        "52W High": w52_col,
                        "BB(50,2)": bb_col,
                        "Signal": rec,
                        "SL Satellite": round(sl_sat, 2),
                        "SL Sat Buffer %": round(sl_sat_buffer, 2),
                        "SL Core": round(sl_core, 2),
                        "SL Core Buffer %": round(sl_core_buffer, 2),
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(results)

# ==========================================
# 2. MAIN DASHBOARD VIEW
# ==========================================
st.title("📊 Shaktimaan Hybrid — Multi-Stock Classification & Stop Loss Dashboard")
st.subheader(f"Active Watchlist: `{selected_wl}`")
st.write(f"Tokens to scan: `{', '.join(tickers)}`")

if st.button("🚀 Run Classification & Stop Loss Scan", type="primary"):
    if not tickers:
        st.warning("Selected watchlist is empty.")
    else:
        with st.spinner(f"Scanning {len(tickers)} stocks for '{selected_wl}'..."):
            df_out = fetch_and_analyze_batch(tuple(tickers), vol_surge_mult)

        if not df_out.empty:
            # --- ROBUST FILTERING CONTROLS ---
            st.markdown("### 🔍 Filter Results")
            f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
            with f_col1:
                class_options = ["CORE+SATELLITE", "CORE", "SATELLITE", "Neither"]
                selected_classes = st.multiselect("Classification", options=class_options, default=class_options)
            with f_col2:
                stage_options = ["Stage 1: Basing", "Stage 2: Advancing", "Stage 3: Distribution", "Stage 4: Declining"]
                selected_stages = st.multiselect("Weekly Stage", options=stage_options, default=stage_options)
            with f_col3:
                search_query = st.text_input("Ticker Search", "").strip().upper()
            with f_col4:
                signal_options = ["BUY", "HOLD / TRIM", "SELL"]
                selected_signal = st.selectbox("Signal Filter", options=["All"] + signal_options)
            with f_col5:
                only_breakouts = st.checkbox("Active Breakouts Only", value=False)

            # Apply robust filtering logic
            filtered_df = df_out.copy()
            if selected_classes:
                filtered_df = filtered_df[filtered_df["Classification"].isin(selected_classes)]
            if selected_stages:
                filtered_df = filtered_df[filtered_df["Weekly Stage"].apply(lambda x: any(s in x for s in selected_stages))]
            if search_query:
                filtered_df = filtered_df[filtered_df["Ticker"].str.contains(search_query)]
            if selected_signal != "All":
                filtered_df = filtered_df[filtered_df["Signal"].str.contains(selected_signal)]
            if only_breakouts:
                filtered_df = filtered_df[
                    (filtered_df["DC (20)"] != "-")
                    | (filtered_df["52W High"] != "-")
                    | (filtered_df["BB(50,2)"] != "-")
                ]

            if filtered_df.empty:
                st.warning("No stocks match the selected filter criteria. Try expanding your filter options.")
            else:
                # Styling functions
                def style_class(val):
                    palette = {
                        "CORE+SATELLITE": "background-color: #008755; color: white; font-weight: bold",
                        "CORE": "background-color: #147341; color: white; font-weight: bold",
                        "SATELLITE": "background-color: #235aa0; color: white; font-weight: bold",
                        "Neither": "background-color: #323232; color: #bbbbbb",
                    }
                    return palette.get(val, "")

                def style_stage(val):
                    if "Stage 2" in val: return "background-color: #00aa50; color: white; font-weight: bold"
                    if "Stage 1" in val: return "background-color: #808080; color: white"
                    if "Stage 3" in val: return "background-color: #dcb400; color: black; font-weight: bold"
                    if "Stage 4" in val: return "background-color: #dc5064; color: white"
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
                    return "background-color: #1e90ff; color: white; font-weight: bold"

                styled = (
                    filtered_df.style.map(style_class, subset=["Classification"])
                    .map(style_stage, subset=["Weekly Stage"])
                    .map(style_dc, subset=["DC (20)"])
                    .map(style_52w, subset=["52W High"])
                    .map(style_bb, subset=["BB(50,2)"])
                    .map(style_signal, subset=["Signal"])
                    .format({
                        "Last Price": "{:.2f}", 
                        "SL Satellite": "{:.2f}", 
                        "SL Sat Buffer %": "{:.2f}%",
                        "SL Core": "{:.2f}",
                        "SL Core Buffer %": "{:.2f}%"
                    })
                )

                st.dataframe(
                    styled, 
                    use_container_width=True, 
                    height=550,
                    column_config={
                        "Chart": st.column_config.LinkColumn(
                            "TradingView Chart",
                            help="Click to open chart on TradingView",
                            display_text=r"symbol=(.*)",
                            pinned=True
                        )
                    }
                )

                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Export '{selected_wl}' Results to CSV",
                    data=csv_bytes,
                    file_name=f"{selected_wl.lower().replace(' ', '_')}_scan.csv",
                    mime="text/csv",
                )
        else:
            st.error("No valid stock data returned.")
