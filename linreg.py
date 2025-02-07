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

# Streamlit Configuration
st.set_page_config(page_title="Market-Making Portfolio Hedging Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS Styling
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

# Sidebar Inputs for Pipeline Parameters
st.sidebar.header("Pipeline Parameters")
frac_diff_d = st.sidebar.slider("Fractional Differencing Parameter (d)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
rolling_window = st.sidebar.number_input("Rolling Window Size for PCA", min_value=10, max_value=100, value=28, step=1)
data_limit = st.sidebar.number_input("Number of Data Points", min_value=30, max_value=500, value=90, step=10)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=6)
ekf_q_scale = st.sidebar.number_input("EKF Q Scale", value=1e-3, format="%.4e")
ekf_r_scale = st.sidebar.number_input("EKF R Scale", value=1e-3, format="%.4e")
bias_penalty = st.sidebar.number_input("Bias Penalty Weight", value=10.0, step=1.0)

# Utility Functions
def fractional_difference_numpy(series, d, thresh=0.01):
    """Applies fractional differencing to a time series."""
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
    """Applies fractional differencing to each column of a DataFrame."""
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
    """Fetches OHLCV data with retry logic."""
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
    """Builds a DataFrame of closing prices for the given symbols."""
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
    """Performs rolling PCA on the data."""
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
    """Smooths PCA betas using an Extended Kalman Filter."""
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

def det_objective(omega, Q, B, Sigma, bias_penalty):
    """Objective function with bias correction."""
    q = Q - np.dot(B, omega)
    residual_mean_penalty = bias_penalty * (np.mean(q) ** 2)
    diag_q = np.diag(q)
    residual_cov = diag_q @ Sigma @ diag_q
    return np.linalg.det(residual_cov) + residual_mean_penalty

async def run_pipeline_async():
    """Runs the entire hedging pipeline asynchronously."""
    exchange = ccxt.kraken({'enableRateLimit': True})
    try:
        markets = await exchange.load_markets()
        logging.info("Connected to Kraken exchange successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Kraken: {e}")
        return None, f"Error connecting to Kraken: {e}"

    all_symbols = list(markets.keys())
    filtered_symbols = [symbol for symbol in all_symbols if symbol.endswith("/USD")]
    price_df = await build_price_dataframe(exchange, filtered_symbols, timeframe, data_limit)
    await exchange.close()

    if price_df.empty:
        return None, "Price DataFrame is empty. Exiting pipeline."

    price_df_fd = apply_frac_diff_fast(price_df, frac_diff_d)
    price_df_fd.dropna(inplace=True)

    if price_df_fd.empty:
        return None, "No usable data after fractional differencing."

    scaler = StandardScaler()
    scaled_prices = scaler.fit_transform(price_df_fd)
    scaled_df = pd.DataFrame(scaled_prices, index=price_df_fd.index, columns=price_df_fd.columns)

    betas = rolling_pca(scaled_df, window=rolling_window, n_components=2)
    if betas.size == 0:
        return None, "No rolling PCA betas computed – insufficient data."

    smoothed_betas = ekf_pca(betas, betas.shape[1], 2, Q_scale=ekf_q_scale, R_scale=ekf_r_scale)
    B_final = smoothed_betas[-1]
    Q_t = scaled_df.iloc[-1].values
    cov_matrix = np.cov(scaled_df.T) + 1e-8 * np.eye(betas.shape[1])

    objective = lambda omega: det_objective(omega, Q_t, B_final, cov_matrix, bias_penalty)
    bounds = [(-1.0, 2.0)] * B_final.shape[1]
    initial_omega = np.full(B_final.shape[1], 0.5)
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
        return res_df, None
    else:
        return None, f"Optimization failed: {result.message}"

def run_pipeline():
    """Wrapper to run the pipeline."""
    return asyncio.run(run_pipeline_async())

# Streamlit Tabs for Layout
tab_learn, tab_model, tab_residuals = st.tabs(["📚 Learn", "📊 Hedging Model", "🔬 Residual Analysis"])

# Learn Tab
with tab_learn:
    st.header("Understanding Hedging with EKF-PCA")
    st.markdown("""
    This application implements a hedging framework using:
    - **Fractional Differencing:** Makes the data stationary.
    - **Rolling PCA:** Extracts dynamic factor loadings (betas).
    - **EKF-PCA:** Smooths the factor loadings using an Extended Kalman Filter.
    - **Hedge Optimization:** Minimizes the determinant of residual covariance.
    """)

# Hedging Model Tab
with tab_model:
    st.header("Run the Hedging Model")
    if st.button("Run Hedging Pipeline"):
        with st.spinner("Running pipeline..."):
            result_df, error_msg = run_pipeline()
        if error_msg:
            st.error(error_msg)
        else:
            st.success("Pipeline completed successfully!")
            st.dataframe(result_df)
            fig_bar = go.Figure(data=[
                go.Bar(name='Original Position', x=result_df['Asset'], y=result_df['Original_Position']),
                go.Bar(name='Hedged Position', x=result_df['Asset'], y=result_df['Hedged_Position']),
                go.Bar(name='Residual Position', x=result_df['Asset'], y=result_df['Residual_Position'])
            ])
            fig_bar.update_layout(barmode='group', title="Asset Positions")
            st.plotly_chart(fig_bar, use_container_width=True)

# Residual Analysis Tab
with tab_residuals:
    st.header("Residual Analysis")
    if st.button("Show Residual Distribution"):
        result_df, error_msg = run_pipeline()
        if error_msg:
            st.error(error_msg)
        else:
            fig_hist = px.histogram(result_df, x='Residual_Position', nbins=20, title="Residual Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
            fig_scatter = px.scatter(result_df, x='Hedged_Position', y='Residual_Position',
                                     title="Hedged vs Residual Positions", trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
