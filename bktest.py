import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.stats import uniform, randint


# Function to fetch stock data
def fetch_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end, auto_adjust=False)
    stock.reset_index(inplace=True)
    return stock


# Function to preprocess data
def preprocess_data(df):
    df.dropna(inplace=True)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df


# Function to add technical indicators
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Upper_BB'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)

    delta = df['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain.flatten()).rolling(window=14).mean()
    avg_loss = pd.Series(loss.flatten()).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['ATR'] = df['High'] - df['Low']
    df['ATR'] = df['ATR'].rolling(window=14).mean()

    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['ADX'] = abs(df['SMA_50'] - df['SMA_200']) / df['ATR']
    return df.dropna()


# Function to create target variable
def create_target_variable(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()


# Function to build and evaluate an optimized XGBoost model
def train_xgboost_model(df):
    features = ['SMA_50', 'SMA_200', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_BB', 'Lower_BB', 'Momentum',
                'ADX']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.5, 1),
        'colsample_bytree': uniform(0.5, 1)
    }

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, scoring='roc_auc', cv=tscv, verbose=1,
                                n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return best_model


# Function to backtest strategy
def backtest_strategy(df, model):
    features = ['SMA_50', 'SMA_200', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_BB', 'Lower_BB', 'Momentum',
                'ADX']
    df['Predicted_Signal'] = model.predict(df[features])
    df['Strategy_Returns'] = df['Predicted_Signal'] * df['Log_Returns']
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Buy_Hold'] = (1 + df['Log_Returns']).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Cumulative_Buy_Hold'], label='Buy & Hold', color='blue')
    plt.plot(df['Date'], df['Cumulative_Strategy'], label='Strategy', color='red')
    plt.title('Strategy vs Buy & Hold Performance')
    plt.legend()
    plt.show()


# Main function
def main():
    ticker = 'AAPL'
    start_date = '2021-01-01'
    end_date = '2024-03-01'
    df = fetch_data(ticker, start_date, end_date)
    df = preprocess_data(df)
    df = add_technical_indicators(df)
    df = create_target_variable(df)
    model = train_xgboost_model(df)
    backtest_strategy(df, model)


if __name__ == "__main__":
    main()
