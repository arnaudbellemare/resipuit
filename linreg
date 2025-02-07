import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ccxt
from datetime import datetime

# Set page config
st.set_page_config(page_title="Kraken Ticker Regression Explorer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better appearance
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

# Title and description
st.title("📈 Kraken Ticker Regression Explorer")
st.markdown("**Powered by ccxt & Streamlit**")
st.markdown("Fetch live OHLCV data from Kraken and explore linear regression on the closing prices over time.")

# Function to fetch Kraken OHLCV data
def fetch_kraken_data(ticker: str, timeframe: str, limit: int):
    exchange = ccxt.kraken()
    bars = exchange.fetch_ohlcv(ticker, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # Convert timestamp from ms to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    # Create a numeric time column (Unix timestamp as float)
    df['NumericTime'] = df['Timestamp'].apply(lambda x: x.timestamp())
    return df

# Sidebar for Kraken data inputs
st.sidebar.header("Kraken Ticker Data")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="BTC/USD")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)
data_limit = st.sidebar.number_input("Number of Data Points", min_value=10, max_value=1000, value=100)

# Fetch data from Kraken
try:
    data = fetch_kraken_data(ticker_symbol, timeframe, data_limit)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Train a linear regression model: Predict 'Close' based on time
model = LinearRegression()
model.fit(data[['NumericTime']], data['Close'])
data['Predicted Close'] = model.predict(data[['NumericTime']])
data['Residual'] = data['Close'] - data['Predicted Close']
mae = mean_absolute_error(data['Close'], data['Predicted Close'])
mse = mean_squared_error(data['Close'], data['Predicted Close'])
r2 = r2_score(data['Close'], data['Predicted Close'])

# Plotting functions
def plot_data_with_residuals(data):
    fig = px.scatter(data, x='Timestamp', y='Close', title='Time vs. Close Price Regression Analysis',
                     labels={'Timestamp': 'Time', 'Close': 'Close Price'}, opacity=0.7)
    # Plot the regression line using sorted data for clarity
    sorted_data = data.sort_values(by='NumericTime')
    fig.add_scatter(x=sorted_data['Timestamp'], y=sorted_data['Predicted Close'], mode='lines', name='Regression Line')
    # Draw residual lines for each point
    for _, row in data.iterrows():
        fig.add_shape(type='line', x0=row['Timestamp'], y0=row['Close'], x1=row['Timestamp'], y1=row['Predicted Close'],
                      line=dict(color='red', width=1))
    fig.update_layout(template="plotly_white")
    return fig

def plot_residuals(data):
    fig = px.scatter(data, x='Predicted Close', y='Residual', 
                     title='Residual Plot',
                     labels={'Predicted Close': 'Predicted Close Price', 'Residual': 'Residual'}, opacity=0.7)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(template="plotly_white")
    return fig

def plot_residual_distribution(data):
    fig = px.histogram(data, x='Residual', title='Distribution of Residuals')
    fig.update_layout(template="plotly_white")
    return fig

# Tabs for organizing the app content
tab1, tab2, tab3, tab4 = st.tabs(["📚 Learn", "📊 Model", "🔬 Residuals", "🧠 Quiz"])

with tab1:
    st.header("Understanding Linear Regression & Residuals")
    st.markdown("""
    <div style="background-color: #e6e6fa; padding: 20px; border-radius: 10px;">
    <h3>What is Linear Regression?</h3>
    <p>Linear regression is a statistical method for modeling the relationship between a dependent variable and an independent variable. In this example:</p>
    <ul>
        <li><strong>Independent Variable:</strong> Time (converted to Unix timestamp)</li>
        <li><strong>Dependent Variable:</strong> Close Price</li>
        <li><strong>Regression Line:</strong> The line that best fits the observed data points</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #fff0f5; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>What are Residuals?</h3>
    <p>Residuals are the differences between the observed closing prices and the predicted values from our regression model. They indicate the prediction error for each data point.</p>
    <ul>
        <li><span class="highlight">Residual = Actual Close - Predicted Close</span></li>
        <li>Analyzing residuals helps assess the performance and assumptions of the model.</li>
        <li>The regression line is obtained by minimizing the sum of squared residuals.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0fff0; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>Key Metrics</h3>
    <ul>
        <li><strong>Mean Absolute Error (MAE):</strong> Average absolute difference between observed and predicted values.</li>
        <li><strong>Mean Squared Error (MSE):</strong> Average of squared differences between observed and predicted values.</li>
        <li><strong>R² Score:</strong> Proportion of variance in the dependent variable explained by the model.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_data_with_residuals(data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regression Equation")
        st.latex(f"\\hat{{Close}} = {model.intercept_:.2f} + {model.coef_[0]:.4f} \\times Time")
        st.caption("Note: Time is represented as a Unix timestamp.")
        
        st.subheader("Model Metrics")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
    
    with st.expander("View Data Table"):
        st.write(data)

with tab3:
    st.header("🔬 Residual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Residual Plot")
        fig_residuals = plot_residuals(data)
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col2:
        st.subheader("Residual Distribution")
        fig_hist = plot_residual_distribution(data)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Interpretation")
    st.write("""
    - The Residual Plot shows the differences between the actual and predicted closing prices.
    - Ideally, residuals should be randomly scattered around zero, indicating no systematic bias.
    - The Residual Distribution should be approximately normal and centered around zero.
    """)

with tab4:
    st.header("Test Your Knowledge")
    
    questions = [
        {
            "question": "What does a residual represent in a regression model?",
            "options": ["The predicted value", "The actual value", "The difference between actual and predicted value"],
            "correct": 2,
            "explanation": "A residual is the difference between the actual observed value and the value predicted by the regression model."
        },
        {
            "question": "What does a high R² score indicate?",
            "options": ["Poor model fit", "Strong relationship between variables", "No relationship between variables"],
            "correct": 1,
            "explanation": "A high R² score (close to 1) indicates that a large proportion of the variance in the dependent variable is predictable from the independent variable."
        },
        {
            "question": "In an ideal residual plot, what pattern should you observe?",
            "options": ["A clear upward trend", "A clear downward trend", "Random scatter around zero"],
            "correct": 2,
            "explanation": "An ideal residual plot shows a random scatter around the zero line, suggesting that the errors are random and not biased."
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
st.sidebar.info("This app fetches Kraken ticker data via ccxt and explores linear regression and residual analysis on closing prices.")
