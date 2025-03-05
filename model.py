import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Function to fetch stock data
def fetch_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end, auto_adjust=False)

    # Flatten column names if multi-index
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = ['_'.join(col).strip() for col in stock.columns]

    # Rename columns to remove '_AAPL' suffix for consistency
    stock.columns = [col.replace('_AAPL', '') for col in stock.columns]

    stock.reset_index(inplace=True)
    print(stock.head())  # Debugging
    return stock

# Function to preprocess data
def preprocess_data(df):
    df.dropna(inplace=True)  # Remove missing values

    # Ensure correct column naming
    if 'Close' not in df.columns:
        raise KeyError("Column 'Close' not found. Ensure correct column names.")

    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)  # Drop initial NaN from log returns
    return df

# Function to add technical indicators
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['ATR'] = df['High'] - df['Low']
    df['ATR'] = df['ATR'].rolling(window=14).mean()

    return df.dropna()

# Function to plot stock data and indicators
def plot_stock_data(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], label='AAPL Price', color='blue')
    plt.plot(df['Date'], df['SMA_50'], label='SMA 50', color='orange', linestyle='dashed')
    plt.plot(df['Date'], df['SMA_200'], label='SMA 200', color='red', linestyle='dashed')
    plt.title('AAPL Stock Price with Moving Averages')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='dashed', color='red')
    plt.axhline(30, linestyle='dashed', color='green')
    plt.title('AAPL RSI Indicator')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['Date'], df['MACD'], label='MACD', color='black')
    plt.plot(df['Date'], df['Signal_Line'], label='Signal Line', color='red', linestyle='dashed')
    plt.title('AAPL MACD Indicator')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(df['Log_Returns'].dropna(), bins=50, kde=True, color='blue')
    plt.title('Distribution of Log Returns')
    plt.show()

# Function to create target variable
def create_target_variable(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 if price goes up, 0 otherwise
    return df.dropna()  # Drop last row with NaN target

# Function to build and evaluate an XGBoost model
def train_xgboost_model(df):
    features = ['SMA_50', 'SMA_200', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'ATR']
    X = df[features]
    y = df['Target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    return model

# Main function
def main():
    ticker = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2024-03-01'

    df = fetch_data(ticker, start_date, end_date)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = create_target_variable(df)

    plot_stock_data(df)

    model = train_xgboost_model(df)

if __name__ == "__main__":
    main()
