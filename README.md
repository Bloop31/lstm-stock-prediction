# LSTM Stock Price Prediction

This project uses a **Long Short-Term Memory (LSTM) neural network** to predict stock prices using historical time-series data.

The model learns patterns from previous stock prices and attempts to forecast future price movements.

---

## Project Overview

Time-series data such as stock prices contain **temporal dependencies**, meaning past values influence future values.

Traditional machine learning models struggle with this type of data, so this project uses **LSTM networks**, which are designed to learn long-term dependencies in sequences.

---

## Features

- Download stock market data using Yahoo Finance
- Data preprocessing and normalization
- Time-series sequence generation
- LSTM neural network for prediction
- Visualization of predicted vs actual prices

---

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- yfinance

---

## How It Works

1. Download stock price data from Yahoo Finance
2. Normalize data using MinMaxScaler
3. Create sliding windows of historical data
4. Train an LSTM neural network
5. Generate predictions
6. Compare predicted vs actual stock prices

---

## Example Output

The model generates a graph comparing:

- Actual stock prices
- Predicted stock prices

This helps visualize how well the LSTM captures market trends.

---

