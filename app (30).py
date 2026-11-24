import math
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Student Investment & Trading Simulator",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------
# 1. CURATED ASSET UNIVERSE BY PORTFOLIO CATEGORY
# ---------------------------------------------------------
CATEGORY_ASSETS = {
    "Core Index ETFs (Broad Market)": {
        "SPDR S&P 500 ETF (SPY)": "SPY",
        "Invesco QQQ Trust (QQQ)": "QQQ",
        "Vanguard Total Stock Market ETF (VTI)": "VTI",
        "Vanguard Dividend Appreciation ETF (VIG)": "VIG",
    },
    "Blue-Chip & Dividend Leaders (Defensive)": {
        "Apple Inc. (AAPL)": "AAPL",
        "Microsoft Corp. (MSFT)": "MSFT",
        "Johnson & Johnson (JNJ)": "JNJ",
        "Coca-Cola Co. (KO)": "KO",
        "JPMorgan Chase & Co. (JPM)": "JPM",
    },
    "High-Growth & Tech Leaders (Growth)": {
        "NVIDIA Corp. (NVDA)": "NVDA",
        "Alphabet Inc. (GOOGL)": "GOOGL",
        "Amazon.com Inc. (AMZN)": "AMZN",
        "Meta Platforms Inc. (META)": "META",
        "Broadcom Inc. (AVGO)": "AVGO",
    },
    "High-Volatility & Active Trading (Aggressive)": {
        "Tesla Inc. (TSLA)": "TSLA",
        "Advanced Micro Devices (AMD)": "AMD",
        "Palantir Technologies (PLTR)": "PLTR",
        "Coinbase Global Inc. (COIN)": "COIN",
    },
}

RISK_PROFILES = {
    "Conservative": {
        "description": "Prioritizes capital preservation and steady dividends with lower volatility.",
        "allocation": {
            "Blue-Chip & Dividend Leaders (Defensive)": 55,
            "Core Index ETFs (Broad Market)": 35,
            "High-Growth & Tech Leaders (Growth)": 10,
            "High-Volatility & Active Trading (Aggressive)": 0,
        },
        "allowed_strategies": ["Buy Stock (Long Equity)", "Sell Cash-Secured Put / Covered Call (Income)"],
    },
    "Moderate": {
        "description": "Balances long-term market growth with defensive blue-chip stability.",
        "allocation": {
            "Core Index ETFs (Broad Market)": 45,
            "Blue-Chip & Dividend Leaders (Defensive)": 30,
            "High-Growth & Tech Leaders (Growth)": 20,
            "High-Volatility & Active Trading (Aggressive)": 5,
        },
        "allowed_strategies": [
            "Buy Stock (Long Equity)",
            "Buy Call Option (Bullish Leverage)",
            "Buy Put Option (Downside Protection)",
            "Sell Cash-Secured Put / Covered Call (Income)",
        ],
    },
    "Growth": {
        "description": "Targets above-average capital appreciation and accepts moderate-to-high swings.",
        "allocation": {
            "High-Growth & Tech Leaders (Growth)": 45,
            "Core Index ETFs (Broad Market)": 30,
            "Blue-Chip & Dividend Leaders (Defensive)": 10,
            "High-Volatility & Active Trading (Aggressive)": 15,
        },
        "allowed_strategies": [
            "Buy Stock (Long Equity)",
            "Buy Call Option (Bullish Leverage)",
            "Buy Put Option (Downside Protection)",
            "Sell Cash-Secured Put / Covered Call (Income)",
            "Short the Stock (Bearish Equity)",
        ],
    },
    "Aggressive": {
        "description": "Maximizes return potential using high-beta stocks, options leverage, and tactical shorting.",
        "allocation": {
            "High-Growth & Tech Leaders (Growth)": 40,
            "High-Volatility & Active Trading (Aggressive)": 35,
            "Core Index ETFs (Broad Market)": 20,
            "Blue-Chip & Dividend Leaders (Defensive)": 5,
        },
        "allowed_strategies": [
            "Buy Stock (Long Equity)",
            "Buy Call Option (Bullish Leverage)",
            "Buy Put Option (Downside Protection)",
            "Sell Cash-Secured Put / Covered Call (Income)",
            "Short the Stock (Bearish Equity)",
        ],
    },
}

# ---------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ---------------------------------------------------------
INITIAL_BUDGET = 10000.0

if "risk_profile" not in st.session_state:
    st.session_state.risk_profile = None
if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []
if "cash" not in st.session_state:
    st.session_state.cash = INITIAL_BUDGET
if "minute_history" not in st.session_state:
    st.session_state.minute_history = []


# ---------------------------------------------------------
# 3. DATA & FINANCIAL HELPER FUNCTIONS
# ---------------------------------------------------------
def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function without external dependencies."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Fallback theoretical option price when live option chains are unavailable after hours."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


@st.cache_data(ttl=55)
def get_live_quote_and_intraday(ticker: str):
    """Fetches latest price and 1-minute intraday bars (cached 55s for 1-minute updates)."""
    stock = yf.Ticker(ticker)
    intraday = pd.DataFrame()
    try:
        intraday = stock.history(period="1d", interval="1m")
        if intraday.empty:
            intraday = stock.history(period="5d", interval="5m")
    except Exception:
        pass

    hist_daily = pd.DataFrame()
    try:
        hist_daily = stock.history(period="1y", interval="1d")
    except Exception:
        pass

    current_price = None
    if not intraday.empty:
        current_price = float(intraday["Close"].iloc[-1])
    elif not hist_daily.empty:
        current_price = float(hist_daily["Close"].iloc[-1])

    # Estimate annualized volatility from daily returns
    ann_vol = 0.28
    if len(hist_daily) > 20:
        daily_ret = hist_daily["Close"].pct_change().dropna()
        ann_vol = float(daily_ret.std() * np.sqrt(252))

    return current_price, intraday, hist_daily, ann_vol


@st.cache_data(ttl=300)
def get_stock_fundamentals_and_cagr(ticker: str):
    """Fetches 3-year historical CAGR per share and key fundamentals."""
    stock = yf.Ticker(ticker)
    hist_3y = pd.DataFrame()
    info = {}
    try:
        hist_3y = stock.history(period="3y", interval="1d")
    except Exception:
        pass
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    cagr = 10.0  # sensible fallback if ticker is newly listed
    if len(hist_3y) > 30:
        start_p = float(hist_3y["Close"].iloc[0])
        end_p = float(hist_3y["Close"].iloc[-1])
        years = max((hist_3y.index[-1] - hist_3y.index[0]).days / 365.25, 0.25)
        if start_p > 0:
            cagr = ((end_p / start_p) ** (1.0 / years) - 1.0) * 100.0

    return cagr, info, hist_3y


@st.cache_data(ttl=300)
def get_option_quote(ticker: str, current_price: float, ann_vol: float, option_type: str = "call"):
    """Fetches nearest ~30-day ATM option quote from Yahoo Finance or computes Black-Scholes fallback."""
    stock = yf.Ticker(ticker)
    strike = round(current_price / 5.0) * 5.0 if current_price > 50 else round(current_price, 1)
    if strike <= 0:
        strike = round(current_price, 2)
    expiry_str = "30-Day Synthetic"
    premium = None

    try:
        expirations = stock.options
        if expirations:
            # Pick expiration roughly 21-45 days out if available, else nearest
            chosen_exp = expirations[min(2, len(expirations) - 1)]
            chain = stock.option_chain(chosen_exp)
            df_opt = chain.calls if option_type == "call" else chain.puts
            if not df_opt.empty:
                df_opt = df_opt.copy()
                df_opt["dist"] = (df_opt["strike"] - current_price).abs()
                atm_row = df_opt.sort_values("dist").iloc[0]
                strike = float(atm_row["strike"])
                bid = float(atm_row.get("bid", 0) or 0)
                ask = float(atm_row.get("ask", 0) or 0)
                last = float(atm_row.get("lastPrice", 0) or 0)
                if bid > 0 and ask > 0:
                    premium = (bid + ask) / 2.0
                elif last > 0:
                    premium = last
                expiry_str = chosen_exp
    except Exception:
        pass

    if premium is None or premium <= 0.05:
        premium = black_scholes_price(
            S=current_price,
            K=strike,
            T=30.0 / 365.0,
            r=0.042,
            sigma=max(ann_vol, 0.15),
            option_type=option_type,
        )

    return round(strike, 2), round(max(premium, 0.10), 2), expiry_str


def compute_minute_signal(intraday: pd.DataFrame):
    """Computes 1-minute technical indicators (SMA5 vs SMA15 + RSI14) and returns actionable feedback."""
    if intraday.empty or len(intraday) < 15:
        return "HOLD", "Insufficient intraday bars yet — waiting for more 1-minute candles.", 50.0, 0.0

    closes = intraday["Close"]
    latest = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])
    pct_1m = ((latest - prev) / prev) * 100.0

    sma5 = float(closes.tail(5).mean())
    sma15 = float(closes.tail(15).mean())

    # 14-period RSI
    delta = closes.diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    if loss == 0:
        rsi = 100.0
    else:
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    if rsi < 32 and latest >= sma5:
        signal = "BUY"
        reason = f"Oversold bounce: 1-min RSI is {rsi:.1f} (<32) and price (${latest:.2f}) crossed above 5-min SMA (${sma5:.2f})."
    elif sma5 > sma15 * 1.0008 and rsi < 68:
        signal = "BUY"
        reason = f"Bullish momentum: 5-min SMA (${sma5:.2f}) is trending above 15-min SMA (${sma15:.2f}) with healthy RSI ({rsi:.1f})."
    elif rsi > 70 and latest <= sma5:
        signal = "SELL"
        reason = f"Overbought pullback: 1-min RSI is {rsi:.1f} (>70) and price slipped below 5-min SMA (${sma5:.2f})."
    elif sma5 < sma15 * 0.9992 and rsi > 35:
        signal = "SELL"
        reason = f"Bearish crossover: 5-min SMA (${sma5:.2f}) dropped below 15-min SMA (${sma15:.2f}); consider trimming or hedging."
    else:
        signal = "HOLD"
        reason = f"Consolidation: Price (${latest:.2f}) is near equilibrium (RSI {rsi:.1f}, 5m SMA ${sma5:.2f} vs 15m SMA ${sma15:.2f})."

    return signal, reason, rsi, pct_1m


# ---------------------------------------------------------
# 4. SIDEBAR: $10,000 PORTFOLIO BUDGET & STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Trading Desk")
    invested_total = INITIAL_BUDGET - st.session_state.cash
    st.metric("Available Cash", f"${st.session_state.cash:,.2f}")
    st.metric("Capital Allocated", f"${invested_total:,.2f} / $10,000.00")
    st.progress(min(invested_total / INITIAL_BUDGET, 1.0))

    if st.session_state.risk_profile:
        st.success(f"**Risk Profile:** {st.session_state.risk_profile}")
    else:
        st.warning("Complete Step 1 (Risk Assessment) first.")

    st.markdown("---")
    if st.button("🔄 Reset $10,000 Portfolio", use_container_width=True):
        st.session_state.portfolio = []
        st.session_state.cash = INITIAL_BUDGET
        st.session_state.minute_history = []
        st.rerun()

# ---------------------------------------------------------
# 5. MAIN NAVIGATION TABS
# ---------------------------------------------------------
st.title("📈 Student Stock, Options & Portfolio Lab")
st.caption("Assess your risk profile, analyze stocks vs. options/short strategies, build a $10,000 portfolio, and monitor live 1-minute signals.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Risk Assessment",
    "2️⃣ Strategy Lab & Trade Builder",
    "3️⃣ $10k Portfolio & Growth Simulator",
    "4️⃣ Live 1-Min Signals & Auto-Monitor",
])

# =========================================================
# TAB 1: STUDENT RISK ASSESSMENT
# =========================================================
with tab1:
    st.header("Step 1: Student Investor Risk Assessment")
    st.write("Answer these 5 questions to determine your investor profile and recommended portfolio mix.")

    with st.form("risk_quiz"):
        q1 = st.radio(
            "1. How long do you plan to keep this $10,000 invested before needing the cash?",
            [
                "Less than 1 year (Short-term tuition / expenses)",
                "1 to 3 years (Near-term goal)",
                "3 to 7 years (Post-graduation wealth building)",
                "7+ years (Long-term compounding)",
            ],
        )
        q2 = st.radio(
            "2. If your portfolio dropped 15% in a single month during a market pullback, what would you do?",
            [
                "Sell everything immediately to prevent further losses",
                "Sell some higher-risk positions and move to cash",
                "Hold steady and wait for recovery",
                "Buy more shares or call options at the discounted price",
            ],
        )
        q3 = st.radio(
            "3. What is your primary objective for this $10,000 simulation?",
            [
                "Protect principal and earn steady dividend income",
                "Grow steadily alongside the overall S&P 500 market",
                "Outperform the market by focusing on high-growth companies",
                "Maximize short-term gains using leverage, options, and active trades",
            ],
        )
        q4 = st.radio(
            "4. How familiar are you with stock options (Calls/Puts) and short selling?",
            [
                "Complete beginner — I only understand buying regular shares",
                "Basic — I know what calls and puts are conceptually",
                "Intermediate — I understand strike prices, premiums, and margin",
                "Advanced — I actively study Greeks, spreads, and short-sale risk",
            ],
        )
        q5 = st.radio(
            "5. How much of your portfolio are you comfortable putting in volatile single stocks?",
            [
                "0% to 10%",
                "10% to 25%",
                "25% to 50%",
                "Over 50%",
            ],
        )

        submitted = st.form_submit_button("Calculate My Risk Profile & Suggested Portfolio")

    if submitted:
        score = 0
        for ans, opts in [
            (q1, ["Less than 1 year", "1 to 3 years", "3 to 7 years", "7+ years"]),
            (q2, ["Sell everything", "Sell some", "Hold steady", "Buy more"]),
            (q3, ["Protect principal", "Grow steadily", "Outperform", "Maximize"]),
            (q4, ["Complete beginner", "Basic", "Intermediate", "Advanced"]),
            (q5, ["0% to 10%", "10% to 25%", "25% to 50%", "Over 50%"]),
        ]:
            for idx, prefix in enumerate(opts):
                if ans.startswith(prefix):
                    score += (idx + 1)

        st.session_state.risk_score = score
        if score <= 8:
            st.session_state.risk_profile = "Conservative"
        elif score <= 12:
            st.session_state.risk_profile = "Moderate"
        elif score <= 16:
            st.session_state.risk_profile = "Growth"
        else:
            st.session_state.risk_profile = "Aggressive"

    if st.session_state.risk_profile:
        prof_name = st.session_state.risk_profile
        prof_data = RISK_PROFILES[prof_name]

        st.markdown("---")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader(f"Your Profile: {prof_name} (Score: {st.session_state.risk_score}/20)")
            st.info(prof_data["description"])

            alloc_df = pd.DataFrame({
                "Portfolio Category": list(prof_data["allocation"].keys()),
                "Suggested Weight (%)": list(prof_data["allocation"].values()),
                "Target Dollar Amount ($)": [INITIAL_BUDGET * (w / 100.0) for w in prof_data["allocation"].values()],
            })
            st.dataframe(
                alloc_df.style.format({"Suggested Weight (%)": "{:.0f}%", "Target Dollar Amount ($)": "${:,.2f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**Recommended Trading Strategies for Your Profile:**")
            for s in prof_data["allowed_strategies"]:
                st.write(f"- {s}")

        with c2:
            fig_alloc = px.pie(
                alloc_df[alloc_df["Suggested Weight (%)"] > 0],
                names="Portfolio Category",
                values="Suggested Weight (%)",
                title=f"{prof_name} Suggested Asset Allocation",
                hole=0.4,
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        st.success("Proceed to **Tab 2: Strategy Lab & Trade Builder** to pick assets from these categories and structure your trades.")

# =========================================================
# TAB 2: STRATEGY LAB & TRADE BUILDER ($10,000 BUDGET)
# =========================================================
with tab2:
    st.header("Step 2: Pick Category Assets & Analyze Trade Strategies")

    active_profile = st.session_state.risk_profile or "Moderate"
    if not st.session_state.risk_profile:
        st.info("Tip: You haven't submitted Tab 1 yet—defaulting to **Moderate** profile suggestions.")

    prof_alloc = RISK_PROFILES[active_profile]["allocation"]
    recommended_cats = [k for k, v in prof_alloc.items() if v > 0]
    all_cats = list(CATEGORY_ASSETS.keys())

    col_a, col_b, col_c = st.columns([1.3, 1.3, 1])
    with col_a:
        chosen_category = st.selectbox(
            "1. Select Portfolio Category (Suggested categories listed first)",
            options=recommended_cats + [c for c in all_cats if c not in recommended_cats],
        )
    with col_b:
        asset_map = CATEGORY_ASSETS[chosen_category]
        chosen_label = st.selectbox("2. Choose an Asset in this Category", list(asset_map.keys()))
        ticker = asset_map[chosen_label]
    with col_c:
        custom_t = st.text_input("Or Override Ticker (Optional):").strip().upper()
        if custom_t:
            ticker = custom_t
            chosen_label = f"Custom ({custom_t})"

    current_price, intraday_df, daily_df, ann_vol = get_live_quote_and_intraday(ticker)
    cagr_3y, info, _ = get_stock_fundamentals_and_cagr(ticker)

    if current_price is None:
        st.error(f"Could not retrieve live pricing for **{ticker}**. Try another ticker.")
    else:
        # Display Live Asset Snapshot
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Live Price", f"${current_price:,.2f}")
        m2.metric("3-Yr Historical CAGR / Share", f"{cagr_3y:.2f}%")
        m3.metric("Annualized Volatility", f"{ann_vol * 100:.1f}%")
        m4.metric("Remaining Cash Budget", f"${st.session_state.cash:,.2f}")

        st.markdown("---")
        strategy = st.selectbox(
            "3. Choose How You Want to Trade This Asset",
            [
                "Buy Stock (Long Equity)",
                "Buy Call Option (Bullish Leverage)",
                "Buy Put Option (Bearish / Downside Hedge)",
                "Sell Cash-Secured Put (Income Strategy)",
                "Short the Stock (Bearish Equity)",
            ],
        )

        # Fetch option parameters if needed
        call_strike, call_prem, call_exp = get_option_quote(ticker, current_price, ann_vol, "call")
        put_strike, put_prem, put_exp = get_option_quote(ticker, current_price, ann_vol, "put")

        # Explain Strategy & Calculate Exact Trade Mechanics
        st.subheader(f"Strategy Breakdown: {strategy} on {ticker}")

        left_col, right_col = st.columns([1.15, 1])
        price_range = np.linspace(current_price * 0.70, current_price * 1.30, 100)

        with left_col:
            if strategy == "Buy Stock (Long Equity)":
                st.markdown(
                    """
                    **What This Means:** You purchase actual shares of the company. You become a partial owner, benefit from any dividends, and profit dollar-for-dollar if the stock rises above your purchase price.
                    - **Max Profit:** Unlimited as the stock price rises.
                    - **Max Loss:** Limited to the amount you invest (if the stock goes to $0).
                    - **Breakeven:** Exact purchase price (**${:.2f}**).
                    """.format(current_price)
                )
                max_shares = int(st.session_state.cash // current_price)
                qty = st.number_input("Number of Shares to Buy", min_value=1, max_value=max(1, max_shares), value=min(5, max(1, max_shares)))
                capital_required = qty * current_price
                breakeven = current_price
                payoffs = (price_range - current_price) * qty

                st.info(
                    f"**Live Price Analysis:** Buying **{qty} shares** of **{ticker}** at **${current_price:,.2f}** requires **${capital_required:,.2f}**. "
                    f"Every +$1.00 move in {ticker} earns you **+${qty:,.2f}**."
                )

            elif strategy == "Buy Call Option (Bullish Leverage)":
                contract_cost = call_prem * 100.0
                breakeven = call_strike + call_prem
                st.markdown(
                    f"""
                    **What This Means:** One Call contract gives you the **right (not obligation)** to buy **100 shares** of {ticker} at the **${call_strike:,.2f} Strike Price** through expiration (**{call_exp}**).
                    - **Option Premium:** ${call_prem:,.2f}/share (**${contract_cost:,.2f}** per 100-share contract).
                    - **Max Profit:** Unlimited above the breakeven price.
                    - **Max Loss:** Strictly capped at the **${contract_cost:,.2f}** premium paid per contract.
                    - **Breakeven at Expiration:** Strike (${call_strike:,.2f}) + Premium (${call_prem:,.2f}) = **${breakeven:,.2f}**.
                    """
                )
                max_contracts = int(st.session_state.cash // contract_cost)
                qty = st.number_input("Number of Call Contracts (1 contract = 100 shares)", min_value=1, max_value=max(1, max_contracts), value=1)
                capital_required = qty * contract_cost
                payoffs = (np.maximum(0.0, price_range - call_strike) - call_prem) * 100.0 * qty

                pct_to_be = ((breakeven - current_price) / current_price) * 100.0
                st.info(
                    f"**Live Price Analysis:** At **${current_price:,.2f}**, {ticker} needs to rise **{pct_to_be:+.2f}%** to reach your **${breakeven:,.2f}** breakeven by {call_exp}. "
                    f"You control **{qty * 100} shares** for **${capital_required:,.2f}** (vs. **${qty * 100 * current_price:,.2f}** to buy the shares outright)."
                )

            elif strategy == "Buy Put Option (Bearish / Downside Hedge)":
                contract_cost = put_prem * 100.0
                breakeven = put_strike - put_prem
                st.markdown(
                    f"""
                    **What This Means:** One Put contract gives you the **right (not obligation)** to sell **100 shares** of {ticker} at the **${put_strike:,.2f} Strike Price** through **{put_exp}**. Traders buy puts to profit from price drops or insure existing shares.
                    - **Option Premium:** ${put_prem:,.2f}/share (**${contract_cost:,.2f}** per contract).
                    - **Max Profit:** Up to ${(put_strike - put_prem) * 100:,.2f} per contract if the stock falls to $0.
                    - **Max Loss:** Capped at the **${contract_cost:,.2f}** premium paid per contract.
                    - **Breakeven at Expiration:** Strike (${put_strike:,.2f}) − Premium (${put_prem:,.2f}) = **${breakeven:,.2f}**.
                    """
                )
                max_contracts = int(st.session_state.cash // contract_cost)
                qty = st.number_input("Number of Put Contracts (1 contract = 100 shares)", min_value=1, max_value=max(1, max_contracts), value=1)
                capital_required = qty * contract_cost
                payoffs = (np.maximum(0.0, put_strike - price_range) - put_prem) * 100.0 * qty

                pct_to_be = ((breakeven - current_price) / current_price) * 100.0
                st.info(
                    f"**Live Price Analysis:** At **${current_price:,.2f}**, {ticker} must fall **{pct_to_be:.2f}%** below **${breakeven:,.2f}** by {put_exp} for this put to turn a net profit. "
                    f"Total upfront cost: **${capital_required:,.2f}**."
                )

            elif strategy == "Sell Cash-Secured Put (Income Strategy)":
                collateral_per_contract = put_strike * 100.0
                credit_per_contract = put_prem * 100.0
                breakeven = put_strike - put_prem
                st.markdown(
                    f"""
                    **What This Means:** You **sell** a Put option and collect **${credit_per_contract:,.2f}** upfront cash income per contract. In exchange, you set aside **${collateral_per_contract:,.2f}** cash collateral to buy 100 shares at **${put_strike:,.2f}** if the stock drops below the strike by **{put_exp}**.
                    - **Max Profit:** Capped at the **${credit_per_contract:,.2f}** premium collected (if {ticker} stays ≥ ${put_strike:,.2f}).
                    - **Max Loss:** Substantial if the stock crashes toward $0 (collateral minus premium kept).
                    - **Breakeven:** **${breakeven:,.2f}** (effective discounted entry price).
                    """
                )
                max_contracts = max(1, int(st.session_state.cash // collateral_per_contract))
                qty = st.number_input("Number of Contracts to Sell", min_value=1, max_value=max_contracts, value=1)
                capital_required = qty * (collateral_per_contract - credit_per_contract)
                payoffs = (put_prem - np.maximum(0.0, put_strike - price_range)) * 100.0 * qty

                st.info(
                    f"**Live Price Analysis:** You immediately collect **${qty * credit_per_contract:,.2f}** in option income, requiring **${capital_required:,.2f}** net cash collateral. "
                    f"Yield on collateral: **{(credit_per_contract / collateral_per_contract) * 100:.2f}%** over ~30 days."
                )

            else:  # Short the Stock (Bearish Equity)
                # Standard Reg-T 50% initial margin requirement
                margin_per_share = current_price * 0.50
                st.markdown(
                    f"""
                    **What This Means:** You borrow shares from your broker and sell them immediately at **${current_price:,.2f}**, hoping to buy them back later at a lower price to pocket the difference.
                    - **Margin Requirement:** Brokers require **50% initial margin** (**${margin_per_share:,.2f}/share**) held as collateral.
                    - **Max Profit:** **${current_price:,.2f}/share** (if the stock drops to $0).
                    - **Max Loss:** **Theoretically Unlimited** if the stock surges upward!
                    - **Breakeven:** **${current_price:,.2f}** (before borrow fees/dividends).
                    """
                )
                max_short_shares = max(1, int(st.session_state.cash // margin_per_share))
                qty = st.number_input("Number of Shares to Short", min_value=1, max_value=max_short_shares, value=min(5, max_short_shares))
                capital_required = qty * margin_per_share
                breakeven = current_price
                payoffs = (current_price - price_range) * qty

                st.warning(
                    f"**Live Price Analysis:** Shorting **{qty} shares** at **${current_price:,.2f}** locks up **${capital_required:,.2f}** in margin collateral. "
                    f"If {ticker} drops 10% to **${current_price * 0.90:,.2f}**, you gain **+${qty * current_price * 0.10:,.2f}**; if it rises 10%, you lose **-${qty * current_price * 0.10:,.2f}**."
                )

        # Right Column: Expiration / Target Price Payoff Diagram
        with right_col:
            fig_payoff = go.Figure()
            fig_payoff.add_trace(
                go.Scatter(x=price_range, y=payoffs, mode="lines", name="Profit / Loss ($)", line=dict(width=3))
            )
            fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_payoff.add_vline(
                x=current_price,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Current: ${current_price:.2f}",
            )
            fig_payoff.update_layout(
                title=f"Profit/Loss Payoff Curve ({ticker})",
                xaxis_title=f"Future {ticker} Share Price ($)",
                yaxis_title="Net Profit / Loss ($)",
                height=380,
            )
            st.plotly_chart(fig_payoff, use_container_width=True)

        # Add Position Button
        if capital_required > st.session_state.cash:
            st.error(
                f"Insufficient cash! This trade requires **${capital_required:,.2f}**, but you only have **${st.session_state.cash:,.2f}** left."
            )
        else:
            if st.button(f"➕ Add Trade to $10,000 Portfolio (${capital_required:,.2f})", type="primary"):
                st.session_state.cash -= capital_required
                st.session_state.portfolio.append({
                    "Ticker": ticker,
                    "Category": chosen_category,
                    "Strategy": strategy,
                    "Quantity": qty,
                    "Entry Price": round(current_price, 2),
                    "Strike": call_strike if "Call" in strategy else (put_strike if "Put" in strategy else None),
                    "Premium": call_prem if "Call" in strategy else (put_prem if "Put" in strategy else None),
                    "Capital Allocated ($)": round(capital_required, 2),
                    "3-Yr CAGR (%)": round(cagr_3y, 2),
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                })
                st.success(f"Added **{strategy}** on **{ticker}** (${capital_required:,.2f})! Remaining cash: **${st.session_state.cash:,.2f}**.")
                st.rerun()

# =========================================================
# TAB 3: $10,000 PORTFOLIO VIEW & GROWTH SIMULATOR
# =========================================================
with tab3:
    st.header("Step 3: Portfolio Overview & Compounding Simulator")

    if not st.session_state.portfolio:
        st.info("Your portfolio is currently empty ($10,000.00 in cash). Go to **Tab 2** to add positions.")
    else:
        port_df = pd.DataFrame(st.session_state.portfolio)

        # Fetch live prices for each holding to compute live P&L and average rate of return per share
        live_prices = []
        unrealized_pnls = []
        for _, row in port_df.iterrows():
            lp, _, _, _ = get_live_quote_and_intraday(row["Ticker"])
            lp = lp if lp is not None else row["Entry Price"]
            live_prices.append(round(lp, 2))

            strat = row["Strategy"]
            qty = row["Quantity"]
            entry = row["Entry Price"]
            if strat == "Buy Stock (Long Equity)":
                pnl = (lp - entry) * qty
            elif strat == "Short the Stock (Bearish Equity)":
                pnl = (entry - lp) * qty
            elif "Call" in strat:
                intrinsic = max(0.0, lp - row["Strike"])
                pnl = (intrinsic - row["Premium"]) * 100.0 * qty
            else:
                intrinsic = max(0.0, row["Strike"] - lp)
                if "Buy Put" in strat:
                    pnl = (intrinsic - row["Premium"]) * 100.0 * qty
                else:
                    pnl = (row["Premium"] - intrinsic) * 100.0 * qty
            unrealized_pnls.append(round(pnl, 2))

        port_df["Live Price ($)"] = live_prices
        port_df["Unrealized P&L ($)"] = unrealized_pnls

        total_allocated = port_df["Capital Allocated ($)"].sum()
        total_pnl = port_df["Unrealized P&L ($)"].sum()
        total_account_value = st.session_state.cash + total_allocated + total_pnl

        # Weighted average historical annual return per share across invested positions
        weighted_cagr = (
            (port_df["Capital Allocated ($)"] * port_df["3-Yr CAGR (%)"]).sum() / total_allocated
            if total_allocated > 0
            else 0.0
        )

        # Key Portfolio Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Portfolio Value", f"${total_account_value:,.2f}", f"${total_pnl:+,.2f}")
        k2.metric("Invested Capital", f"${total_allocated:,.2f}", f"{(total_allocated / INITIAL_BUDGET) * 100:.1f}% Deployed")
        k3.metric("Cash Remaining", f"${st.session_state.cash:,.2f}")
        k4.metric("Avg Annual Return / Share (Weighted)", f"{weighted_cagr:.2f}%")

        st.subheader("Current Holdings & Per-Share Return Breakdown")
        st.dataframe(port_df, use_container_width=True, hide_index=True)

        c_left, c_right = st.columns(2)
        with c_left:
            fig_cat = px.pie(
                port_df,
                names="Category",
                values="Capital Allocated ($)",
                title="Actual Portfolio Allocation by Category",
                hole=0.35,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with c_right:
            fig_strat = px.bar(
                port_df,
                x="Ticker",
                y="3-Yr CAGR (%)",
                color="Strategy",
                title="Historical Average Annual Return per Share (3-Yr CAGR)",
            )
            st.plotly_chart(fig_strat, use_container_width=True)

        # Investment Growth Simulator
        st.markdown("---")
        st.subheader("🔮 What-If Investment Growth Simulator")
        st.write("Explore how different investment amounts would grow over time at your portfolio's weighted average annual return.")

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_principal = st.number_input(
                "Simulated Investment Amount ($)",
                min_value=500,
                max_value=500000,
                value=10000,
                step=1000,
            )
        with sim_c2:
            sim_monthly = st.number_input(
                "Optional Monthly Contribution ($)",
                min_value=0,
                max_value=10000,
                value=0,
                step=100,
            )
        with sim_c3:
            sim_years = st.slider("Time Horizon (Years)", min_value=1, max_value=30, value=10)

        rate = weighted_cagr / 100.0
        years_list = list(range(0, sim_years + 1))
        values_list = []
        contributions_list = []

        for y in years_list:
            fv_lump = sim_principal * ((1.0 + rate) ** y)
            if rate != 0:
                fv_contrib = sim_monthly * 12.0 * (((1.0 + rate) ** y - 1.0) / rate)
            else:
                fv_contrib = sim_monthly * 12.0 * y
            values_list.append(fv_lump + fv_contrib)
            contributions_list.append(sim_principal + sim_monthly * 12.0 * y)

        sim_df = pd.DataFrame({
            "Year": years_list,
            "Projected Portfolio Value ($)": values_list,
            "Total Cash Contributed ($)": contributions_list,
        })

        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Scatter(
                x=sim_df["Year"],
                y=sim_df["Projected Portfolio Value ($)"],
                mode="lines+markers",
                name=f"Projected Value ({weighted_cagr:.2f}%/yr)",
            )
        )
        fig_growth.add_trace(
            go.Scatter(
                x=sim_df["Year"],
                y=sim_df["Total Cash Contributed ($)"],
                mode="lines",
                name="Total Principal Contributed",
                line=dict(dash="dash"),
            )
        )
        fig_growth.update_layout(
            title=f"Projected Growth of ${sim_principal:,.0f} Over {sim_years} Years",
            xaxis_title="Years",
            yaxis_title="Value ($)",
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        st.success(
            f"At **{weighted_cagr:.2f}%** annual return, **${sim_principal:,.2f}** "
            f"(+ **${sim_monthly:,.0f}/mo**) grows to **${values_list[-1]:,.2f}** in **{sim_years} years** "
            f"(Total Investment Gain: **${values_list[-1] - contributions_list[-1]:,.2f}**)."
        )

# =========================================================
# TAB 4: LIVE 1-MINUTE UPDATES & ON-THE-MINUTE FEEDBACK
# =========================================================
with tab4:
    st.header("Step 4: Live 1-Minute Market Feed & Trade Signals")
    st.caption("This section automatically refreshes every 60 seconds using live 1-minute candles (5-min vs. 15-min SMA & 14-min RSI).")

    @st.fragment(run_every=60)
    def live_minute_monitor():
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**Last Minute-by-Minute Scan:** `{now_str}` *(Auto-updates every 60 seconds)*")

        # Determine tickers to monitor: student's portfolio tickers + optional watchlist ticker
        portfolio_tickers = sorted(list({item["Ticker"] for item in st.session_state.portfolio}))
        watch_ticker = st.text_input("Add a Ticker to Live 1-Min Signal Scanner:", value="SPY").strip().upper()

        tickers_to_scan = portfolio_tickers.copy()
        if watch_ticker and watch_ticker not in tickers_to_scan:
            tickers_to_scan.append(watch_ticker)

        if not tickers_to_scan:
            tickers_to_scan = ["SPY", "AAPL"]

        signal_rows = []
        for sym in tickers_to_scan:
            price, intraday, _, _ = get_live_quote_and_intraday(sym)
            if price is not None:
                sig, reason, rsi, pct_1m = compute_minute_signal(intraday)
                in_port = "Yes" if sym in portfolio_tickers else "Watchlist"
                signal_rows.append({
                    "Ticker": sym,
                    "In Portfolio": in_port,
                    "Live Price ($)": round(price, 2),
                    "1-Min Change (%)": round(pct_1m, 3),
                    "1-Min RSI (14)": round(rsi, 1),
                    "On-the-Minute Signal": sig,
                    "Signal Rationale": reason,
                })

        if signal_rows:
            sig_df = pd.DataFrame(signal_rows)

            # Highlight actionable alerts at the top
            for r in signal_rows:
                badge = "🟢 BUY" if r["On-the-Minute Signal"] == "BUY" else ("🔴 SELL" if r["On-the-Minute Signal"] == "SELL" else "🟡 HOLD")
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([1, 1.2, 1, 3.5])
                    c1.markdown(f"### {r['Ticker']}")
                    c2.metric("Live Price", f"${r['Live Price ($)']:.2f}", f"{r['1-Min Change (%)']:+.3f}%")
                    c3.markdown(f"**Action:**\n\n{badge}")
                    c4.write(f"**Minute Feedback:** {r['Signal Rationale']}")

            # Plot 1-minute intraday chart for the first scanned ticker
            focus_sym = tickers_to_scan[0]
            _, focus_intra, _, _ = get_live_quote_and_intraday(focus_sym)
            if not focus_intra.empty:
                focus_intra = focus_intra.copy()
                focus_intra["SMA_5"] = focus_intra["Close"].rolling(5).mean()
                focus_intra["SMA_15"] = focus_intra["Close"].rolling(15).mean()

                fig_live = go.Figure()
                fig_live.add_trace(
                    go.Scatter(x=focus_intra.index, y=focus_intra["Close"], mode="lines", name="1-Min Close")
                )
                fig_live.add_trace(
                    go.Scatter(x=focus_intra.index, y=focus_intra["SMA_5"], mode="lines", name="5-Min SMA", line=dict(dash="dot"))
                )
                fig_live.add_trace(
                    go.Scatter(x=focus_intra.index, y=focus_intra["SMA_15"], mode="lines", name="15-Min SMA", line=dict(dash="dash"))
                )
                fig_live.update_layout(
                    title=f"{focus_sym} Intraday 1-Minute Price & Moving Averages",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    height=360,
                )
                st.plotly_chart(fig_live, use_container_width=True)

            st.caption("Educational Disclaimer: Minute-by-minute technical signals reflect short-term momentum indicators for classroom simulation and are not financial advice.")

    live_minute_monitor()
