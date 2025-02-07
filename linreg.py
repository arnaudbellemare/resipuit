#!/usr/bin/env python3
import streamlit as st
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    filename="portfolio_optimization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set page configuration and custom CSS for styling
st.set_page_config(page_title="Market-Making Portfolio Hedging Explorer", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
.stButton>button {
    background-color: #4b0082;
    color: white;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #e6e6fa;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #8a2be2;
    color: white;
}
.highlight {
    background-color: #ffd700;
    padding: 5px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar inputs for pipeline parameters
st.sidebar.header("Pipeline Parameters")
frac_diff_d = st.sidebar.slider("Fractional Differencing Parameter (d)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
rolling_window = st.sidebar.number_input("Rolling Window Size for PCA", min_value=10, max_value=100, value=28, step=1)
data_limit = st.sidebar.number_input("Number of Data Points", min_value=30, max_value=500, value=90, step=10)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=6)
ekf_q_scale = st.sidebar.number_input("EKF Q Scale", value=1e-3, format="%.4e")
ekf_r_scale = st.sidebar.number_input("EKF R Scale", value=1e-3, format="%.4e")

# -------------------------------
# Utility Functions
# -------------------------------

def fractional_difference_numpy(series, d, thresh=0.01):
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    len_w = len(w)
    diff_series_length = len(series) - len_w + 1
    if diff_series_length <= 0:
        return np.full(len(series), np.nan)
    shape = (diff_series_length, len_w)
    strides = (series.strides[0], series.strides[0])
    windowed = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
    diff_series = np.dot(windowed, w)
    nan_padding = np.full(len(series) - len(diff_series), np.nan)
    return np.concatenate((nan_padding, diff_series))

def apply_frac_diff_fast(df, d):
    frac_diff_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col].values
        if np.isnan(series).all():
            frac_diff_df[col] = np.nan
            logging.warning(f"All values are NaN for {col}.")
        else:
            non_na = ~np.isnan(series)
            frac_series = fractional_difference_numpy(series[non_na], d, thresh=0.01)
            full_series = np.full_like(series, np.nan, dtype=np.float64)
            full_series[non_na] = frac_series
            frac_diff_df[col] = full_series
    return frac_diff_df

async def fetch_ohlcv_with_retry(exchange, symbol, timeframe='1d', limit=90, max_retries=3, backoff_factor=1.0):
    for attempt in range(1, max_retries + 1):
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}.")
            return df['close']
        except Exception as e:
            logging.warning(f"Attempt {attempt} - Error fetching {symbol}: {e}")
            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)
            else:
                logging.error(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
                return None

async def build_price_dataframe(exchange, symbols, timeframe='1d', limit=90):
    tasks = [fetch_ohlcv_with_retry(exchange, symbol, timeframe, limit) for symbol in symbols]
    close_prices = await asyncio.gather(*tasks)
    price_data = {}
    for symbol, close in zip(symbols, close_prices):
        if close is not None:
            price_data[symbol] = close
        else:
            logging.warning(f"Skipping {symbol} due to insufficient data.")
    price_df = pd.DataFrame(price_data)
    price_df.dropna(axis=1, inplace=True)
    return price_df

def rolling_pca(df, window=28, n_components=2):
    pca = PCA(n_components=n_components)
    betas = []
    for t in range(window, len(df)):
        window_data = df.iloc[t-window:t]
        pca.fit(window_data)
        betas.append(pca.components_.T)
    if len(betas) == 0:
        return np.array([])
    return np.array(betas)

def ekf_pca(betas, n_assets, n_components, Q_scale=1e-3, R_scale=1e-3):
    num_steps = betas.shape[0]
    d_state = n_assets * n_components
    x = betas[0].flatten()
    P = np.eye(d_state) * 1e-2
    Q = np.eye(d_state) * Q_scale
    R = np.eye(d_state) * R_scale
    b_hat = np.empty_like(betas)
    b_hat[0] = betas[0]
    for t in range(1, num_steps):
        x_pred = x
        P_pred = P + Q
        z = betas[t].flatten()
        y = z - x_pred
        try:
            S = P_pred + R
            K = P_pred @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logging.error("EKF-PCA - Singular matrix encountered during inversion.")
            K = np.zeros_like(P_pred)
        x = x_pred + K @ y
        P = (np.eye(d_state) - K) @ P_pred
        b_hat[t] = x.reshape(n_assets, n_components)
    return b_hat

def det_objective(omega, Q, B, Sigma):
    q = Q - np.dot(B, omega)
    diag_q = np.diag(q)
    residual_cov = diag_q @ Sigma @ diag_q
    return np.linalg.det(residual_cov)

# -------------------------------
# Pipeline Async Function
# -------------------------------

async def run_pipeline_async():
    # Connect to Kraken exchange
    exchange = ccxt.kraken({'enableRateLimit': True})
    try:
        markets = await exchange.load_markets()
        logging.info("Connected to Kraken exchange successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Kraken: {e}")
        return None, f"Error connecting to Kraken: {e}"
    
    # Filter symbols: Use USD pairs and exclude specific tokens
    all_symbols = list(markets.keys())
    desired_quote_currencies = ["USD"]
    unwanted_substrings = ['WBTC', 'USDC', 'BUSD', 'TUSD', 'DAI', 'UST', 'AUD', 'GBP', 'EUR']
    filtered_symbols = [
        symbol for symbol in all_symbols
        if any(symbol.endswith(f'/{q}') for q in desired_quote_currencies)
           and not any(excl in symbol for excl in unwanted_substrings)
    ]
    logging.info(f"Symbols after filtering: {len(filtered_symbols)}")
    
    price_df = await build_price_dataframe(exchange, filtered_symbols, timeframe=timeframe, limit=data_limit)
    await exchange.close()
    
    if price_df.empty:
        logging.error("Price DataFrame is empty. Exiting pipeline.")
        return None, "Price DataFrame is empty. Exiting pipeline."
    
    # Apply fractional differencing
    price_df_fd = apply_frac_diff_fast(price_df, frac_diff_d)
    price_df_fd.dropna(inplace=True)
    if price_df_fd.empty:
        logging.error("No usable data after fractional differencing. Exiting pipeline.")
        return None, "No usable data after fractional differencing."
    
    # Standardize data
    scaler = StandardScaler()
    scaled_prices = scaler.fit_transform(price_df_fd)
    scaled_df = pd.DataFrame(scaled_prices, index=price_df_fd.index, columns=price_df_fd.columns)
    logging.info("Data scaling completed.")
    
    # Rolling PCA
    betas = rolling_pca(scaled_df, window=rolling_window, n_components=2)
    if betas.size == 0:
        logging.error("No rolling PCA betas computed – insufficient data.")
        return None, "No rolling PCA betas computed – insufficient data."
    
    n_assets = betas.shape[1]
    smoothed_betas = ekf_pca(betas, n_assets, 2, Q_scale=ekf_q_scale, R_scale=ekf_r_scale)
    B_final = smoothed_betas[-1]
    
    Q_t = scaled_df.iloc[-1].values
    cov_matrix = np.cov(scaled_df.T) + 1e-8 * np.eye(n_assets)
    
    objective = lambda omega: det_objective(omega, Q_t, B_final, cov_matrix)
    bounds = [(0.0, 1.0)] * 2
    initial_omega = np.full(2, 0.5)
    result = minimize(objective, initial_omega, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        optimal_omega = result.x
        residual_q = Q_t - np.dot(B_final, optimal_omega)
        res_df = pd.DataFrame({
            'Asset': scaled_df.columns,
            'Original_Position': Q_t,
            'Hedged_Position': np.dot(B_final, optimal_omega),
            'Residual_Position': residual_q
        })
        res_df['Abs_Residual'] = res_df['Residual_Position'].abs()
        logging.info("Portfolio hedging optimization completed successfully.")
        return res_df, None
    else:
        logging.error(f"Optimization failed: {result.message}")
        return None, f"Optimization failed: {result.message}"

def run_pipeline():
    return asyncio.run(run_pipeline_async())

# -------------------------------
# App Tabs and Layout
# -------------------------------

tab_learn, tab_model, tab_residuals, tab_quiz = st.tabs(["📚 Learn", "📊 Hedging Model", "🔬 Residual Analysis", "🧠 Quiz"])

with tab_learn:
    st.header("Understanding Market-Making Portfolio Hedging with EKF-PCA")
    st.markdown("""
    This application implements a portfolio hedging framework that involves:
    
    - **Fractional Differencing:** Stationarizes asset price series while preserving memory.
    - **Rolling PCA:** Captures time-varying factor loadings (betas) over a rolling window.
    - **EKF-PCA:** Uses an Extended Kalman Filter to smooth the PCA betas.
    - **Hedge Optimization:** Finds optimal hedge ratios by minimizing the determinant of the residual covariance.
    
    Together, these techniques help adjust asset positions dynamically to mitigate portfolio risk.
    """)

with tab_model:
    st.header("Portfolio Hedging Model")
    if st.button("Run Hedging Pipeline"):
        with st.spinner("Running pipeline, please wait..."):
            result_df, error_msg = run_pipeline()
        if error_msg:
            st.error(error_msg)
        else:
            st.success("Hedging pipeline completed successfully!")
            st.subheader("Optimization Results")
            st.dataframe(result_df)
            
            # Bar chart for asset positions
            fig_bar = go.Figure(data=[
                go.Bar(name='Original Position', x=result_df['Asset'], y=result_df['Original_Position']),
                go.Bar(name='Hedged Position', x=result_df['Asset'], y=result_df['Hedged_Position']),
                go.Bar(name='Residual Position', x=result_df['Asset'], y=result_df['Residual_Position'])
            ])
            fig_bar.update_layout(barmode='group', title="Asset Positions")
            st.plotly_chart(fig_bar, use_container_width=True)

with tab_residuals:
    st.header("Residual Analysis")
    st.markdown("Residuals represent the differences between the original and hedged positions. Analyzing these helps assess the effectiveness of the hedge.")
    if st.button("Show Residual Distribution"):
        # For demonstration, run the pipeline again (in a real app, cache the results)
        result_df, error_msg = run_pipeline()
        if error_msg:
            st.error(error_msg)
        else:
            fig_hist = px.histogram(result_df, x='Residual_Position', nbins=20, title="Residual Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            fig_scatter = px.scatter(result_df, x='Hedged_Position', y='Residual_Position', 
                                     title="Hedged vs Residual Positions", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)

with tab_quiz:
    st.header("Test Your Knowledge on Hedging Strategies")
    questions = [
        {
            "question": "What is the purpose of fractional differencing in this pipeline?",
            "options": [
                "To stationarize the price series while preserving memory",
                "To remove noise from the data",
                "To compute a moving average"
            ],
            "correct": 0,
            "explanation": "Fractional differencing is used to make a non-stationary series stationary while preserving the series' memory."
        },
        {
            "question": "What does EKF-PCA accomplish in this hedging framework?",
            "options": [
                "It forecasts future prices",
                "It smooths the time-varying beta estimates from rolling PCA",
                "It calculates the optimal hedge ratio directly"
            ],
            "correct": 1,
            "explanation": "EKF-PCA applies an Extended Kalman Filter to smooth the PCA beta estimates, reducing noise in the dynamic factors."
        },
        {
            "question": "In the hedge optimization step, what is minimized?",
            "options": [
                "The sum of squared errors",
                "The determinant of the residual covariance matrix",
                "The absolute residuals"
            ],
            "correct": 1,
            "explanation": "The optimization minimizes the determinant of the residual covariance matrix to determine the optimal hedge ratios."
        }
    ]
    
    for i, q in enumerate(questions):
        st.subheader(f"Question {i+1}: {q['question']}")
        user_answer = st.radio(f"Select your answer for Question {i+1}:", q['options'], key=f"q{i}")
        if st.button(f"Check Answer for Question {i+1}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("Correct! Great job!")
            else:
                st.error("Not quite. Let's learn from this!")
            st.info(f"Explanation: {q['explanation']}")
        st.write("---")

st.sidebar.markdown("---")
st.sidebar.info("This app fetches market data via Kraken using ccxt, processes it through fractional differencing, rolling PCA, and EKF-PCA, and performs hedge optimization to adjust portfolio positions.")
