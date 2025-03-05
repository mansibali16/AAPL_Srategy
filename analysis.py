import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fetch_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # Reset index to keep 'Date' as a column
    stock.reset_index(inplace=True)

    # Flatten MultiIndex if it exists
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock.columns]

    # Rename columns to standardize
    stock.columns = [col.replace(f"_{ticker}", "") for col in stock.columns]  # Remove "_AAPL"
    stock.rename(columns={"Adj Close": "AdjClose"}, inplace=True)  # No spaces in names

    # Ensure Date is a datetime type and set as index
    stock['Date_'] = pd.to_datetime(stock['Date_'])
    stock.set_index('Date_', inplace=True)

    print(stock.head())  # Debugging output
    return stock


def preprocess_data(df):
    df.dropna(inplace=True)  # Remove missing values

    # Ensure the correct column names exist
    required_columns = ['Close', 'High', 'Low', 'Open']
    for col in required_columns:
        if col not in df.columns:
            print(f"Columns in DataFrame: {df.columns}")
            raise KeyError(f"Column '{col}' not found. Ensure correct column names.")

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)  # Drop initial NaN from log returns
    return df


def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR Calculation (Using True Range)
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift()),
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    return df.dropna()


def plot_stock_data(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='AAPL Price', color='blue')
    plt.plot(df.index, df['SMA_50'], label='SMA 50', color='orange', linestyle='dashed')
    plt.plot(df.index, df['SMA_200'], label='SMA 200', color='red', linestyle='dashed')
    plt.title('AAPL Stock Price with Moving Averages')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red')
    plt.axhline(30, linestyle='dashed', color='green')
    plt.title('AAPL RSI Indicator')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df.index, df['MACD'], label='MACD', color='black')
    plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='red', linestyle='dashed')
    plt.title('AAPL MACD Indicator')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['Log_Returns'].dropna(), bins=50, kde=True, color='blue')
    plt.title('Distribution of Log Returns')
    plt.show()


def main():
    ticker = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2024-03-01'

    df = fetch_data(ticker, start_date, end_date)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    plot_stock_data(df)


if __name__ == "__main__":
    main()




