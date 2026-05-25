# Main Application Module — Stock Prediction App

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data import fetch_stock_data, get_stock_info
from features import add_technical_indicators
from model import (
    prepare_data,
    train_linear_regression,
    train_random_forest,
    evaluate_model,
    predict_next_day,
    get_buy_sell_signal
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- RESPONSIVE CSS ----------------
st.markdown("""
<style>

/* Main App Container */
.block-container {
    max-width: 100%;
    padding-top: 0.7rem;
    padding-bottom: 0.7rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Fix overall scaling */
html, body, [class*="css"] {
    font-size: 14px;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #4CAF50;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Cards */
.card {
    background: #1E1E1E;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #2E2E2E;
}

.card h4 {
    font-size: 15px;
    margin-bottom: 8px;
}

.card h2 {
    font-size: 24px;
    margin: 0;
}

/* Signal Box */
.signal {
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: white;
    margin-top: 12px;
}

/* Info Box */
.box {
    background: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    margin-top: 12px;
    border: 1px solid #2E2E2E;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    width: 260px !important;
}

/* Remove excess spacing */
.element-container {
    margin-bottom: 0.3rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 15px;
}

/* Charts */
.js-plotly-plot {
    width: 100% !important;
}

/* Mobile */
@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .main-title {
        font-size: 26px;
    }

    .card h2 {
        font-size: 20px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<div class='main-title'>📈 Stock Market Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
stocks = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Apple": "AAPL",
    "Tesla": "TSLA"
}

stock = st.sidebar.selectbox(
    "📌 Select Stock",
    list(stocks.keys())
)

ticker = stocks[stock]

period = st.sidebar.selectbox(
    "📅 Select Period",
    ["1y", "2y", "5y"]
)

model_choice = st.sidebar.radio(
    "🤖 Select Model",
    ["Linear Regression", "Random Forest", "Both"]
)

run = st.sidebar.button("🚀 Run Prediction")

# ---------------- UI FUNCTIONS ----------------
def card(title, value):

    st.markdown(
        f"""
        <div class='card'>
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


def signal_box(signal):

    colors = {
        "🟢 BUY": "#4CAF50",
        "🔴 SELL": "#F44336",
        "🟡 HOLD": "#FFC107"
    }

    st.markdown(
        f"""
        <div class='signal'
        style='background:{colors[signal]}'>
        {signal}
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_conclusion(curr, pred, results):

    pct = ((pred - curr) / curr) * 100

    if pct > 1.5:
        trend, action = "Bullish 📈", "BUY"

    elif pct < -1.5:
        trend, action = "Bearish 📉", "SELL"

    else:
        trend, action = "Sideways ➡️", "HOLD"

    best = max(
        results.items(),
        key=lambda x: x[1]['R2']
    )

    model_name = best[0]
    r2 = best[1]['R2']

    conf = (
        "High"
        if r2 > 0.85
        else "Moderate"
        if r2 > 0.65
        else "Low"
    )

    return f"""
### 📊 Prediction Summary

- Change: **{pct:.2f}%**
- Trend: **{trend}**
- Action: **{action}**

### 🤖 Model Insight

- Best Model: **{model_name}**
- R² Score: **{r2}**
- Confidence: **{conf}**
"""


def quick_insights(signal, rsi):

    if rsi > 70:
        momentum = "Overbought ⚠️"

    elif rsi < 30:
        momentum = "Oversold 🔥"

    else:
        momentum = "Neutral"

    return f"""
### ⚡ Quick Insights

- Signal: **{signal}**
- RSI Momentum: **{momentum}**
"""

# ---------------- MAIN ----------------
if run:

    # Fetch stock data
    df = fetch_stock_data(ticker, period)

    # Company info
    info = get_stock_info(ticker)

    # Add technical indicators
    df = add_technical_indicators(df)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # ---------------- MODEL TRAINING ----------------
    if model_choice == "Linear Regression":

        model = train_linear_regression(
            X_train,
            y_train
        )

        results = {
            "Linear Regression": evaluate_model(
                model,
                X_test,
                y_test
            )
        }

        next_day = predict_next_day(model, df)

    elif model_choice == "Random Forest":

        model = train_random_forest(
            X_train,
            y_train
        )

        results = {
            "Random Forest": evaluate_model(
                model,
                X_test,
                y_test
            )
        }

        next_day = predict_next_day(model, df)

    else:

        lr = train_linear_regression(
            X_train,
            y_train
        )

        rf = train_random_forest(
            X_train,
            y_train
        )

        results = {
            "Linear Regression": evaluate_model(
                lr,
                X_test,
                y_test
            ),

            "Random Forest": evaluate_model(
                rf,
                X_test,
                y_test
            )
        }

        next_day = predict_next_day(rf, df)

    # ---------------- CURRENT PRICE ----------------
    current = (
        float(info['current_price'])
        if info['current_price'] != 'N/A'
        else next_day
    )

    signal = get_buy_sell_signal(
        current,
        next_day
    )

    # ---------------- COMPANY INFO ----------------
    st.markdown(f"## 🏢 {info['name']}")
    st.write(f"**Sector:** {info['sector']}")

    # ---------------- MINI TREND ----------------
    st.subheader("📈 Last 30 Days Trend")

    st.line_chart(
        df['Close'].tail(30),
        height=180,
        use_container_width=True
    )

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "📉 Charts",
        "🤖 Models"
    ])

    # ---------------- OVERVIEW ----------------
    with tab1:

        c1, c2 = st.columns(2)

        with c1:
            card(
                "Current Price",
                f"{current:.2f}"
            )

        with c2:
            card(
                "Predicted Price",
                f"{next_day:.2f}"
            )

        signal_box(signal)

        st.markdown(
            f"""
            <div class='box'>
            {generate_conclusion(current, next_day, results)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            quick_insights(
                signal,
                df['RSI'].iloc[-1]
            )
        )

        # Alerts
        if df['RSI'].iloc[-1] > 70:
            st.warning("⚠️ Stock may be overbought")

        elif df['RSI'].iloc[-1] < 30:
            st.success("🔥 Stock may be oversold")

    # ---------------- CHARTS ----------------
    with tab2:

        # Candlestick Chart
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA7'],
                name="MA7"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA21'],
                name="MA21"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA50'],
                name="MA50"
            )
        )

        fig.update_layout(
            title="Stock Price Chart",
            template="plotly_dark",
            height=420,
            autosize=True,
            margin=dict(
                l=10,
                r=10,
                t=40,
                b=10
            ),
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Volume Chart
        vol = go.Figure()

        vol.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume"
            )
        )

        vol.update_layout(
            title="Trading Volume",
            template="plotly_dark",
            height=300,
            autosize=True,
            margin=dict(
                l=10,
                r=10,
                t=40,
                b=10
            )
        )

        st.plotly_chart(
            vol,
            use_container_width=True
        )

    # ---------------- MODELS ----------------
    with tab3:

        for name, res in results.items():

            model_fig = go.Figure()

            model_fig.add_trace(
                go.Scatter(
                    y=res['actuals'],
                    name="Actual"
                )
            )

            model_fig.add_trace(
                go.Scatter(
                    y=res['predictions'],
                    name="Predicted"
                )
            )

            model_fig.update_layout(
                title=f"{name} Performance",
                template="plotly_dark",
                height=350,
                autosize=True,
                margin=dict(
                    l=10,
                    r=10,
                    t=40,
                    b=10
                )
            )

            st.plotly_chart(
                model_fig,
                use_container_width=True
            )

        # Metrics Table
        st.subheader("📋 Model Evaluation")

        metrics_df = pd.DataFrame(results).T[
            ['MAE', 'RMSE', 'R2']
        ]

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

        # Explanation
        st.markdown("""
### 🧠 Why this prediction?

- Uses Moving Averages
- Uses RSI Momentum
- Uses Bollinger Bands
- Uses Volatility Analysis
- Uses Historical Price Trends
""")

# ---------------- EMPTY STATE ----------------
else:

    st.info(
        "👈 Select stock and click 'Run Prediction'"
    )