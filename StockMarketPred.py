import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LSTM    
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker




#Gather the desired stock financial data from yfinance API
Symobl = input("Stock Symbol: ")
symbolTicker = yf.Ticker(Symobl)
incStat = symbolTicker.get_income_stmt().transpose()
totalRevenue = incStat['TotalRevenue']
datesIncStat = totalRevenue.index
datesIncStat = [str(date.date()) for date in datesIncStat]
data = yf.download(Symobl, period="max")

def get_company_name(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.info.get("displayName", "Unknown Company")

Company_name = get_company_name(Symobl)

print("Company Name: ", Company_name)


#Global Variables
prediction_days = 100
future_prediction_days = 14 
now = date.today().strftime('%Y-%m-%d')

#Data Preprocessing
y = data['Close']
x = data.drop(['Close'], axis=1)



def get_last_valid_trading_day(target_date_str, df):
    """
    Finds the most recent valid day in the dataframe's index
    on or before the target date.
    """
    target_date = pd.to_datetime(target_date_str)
    subset = df.loc[:target_date]
    if not subset.empty:
        return subset.index[-1]
    return None

def create_windowed_dataset(dataframe, start_date_str, end_date_str, n=3, feature_col='Close'):
    """
    Creates a windowed dataset for time series forecasting directly into NumPy arrays.
    
    This function is more robust, efficient, and readable than the original.
    """
    # 1. Use robust helper function to find valid start and end dates
    start_date = get_last_valid_trading_day(start_date_str, dataframe)
    end_date = get_last_valid_trading_day(end_date_str, dataframe)
    
    if start_date is None or end_date is None:
        raise ValueError("Could not find valid start or end dates in the dataframe.")
        
    # 2. Filter the dataframe to the relevant date range ONCE
    relevant_data = dataframe.loc[start_date:end_date, feature_col]
    
    dates = []
    X, Y = [], []

    # 3. Iterate efficiently over the pre-filtered data
    # We start from index 'n' because we need 'n' previous days of history
    for i in range(n, len(relevant_data)):
        # Use efficient integer-based slicing (.iloc) on the series
        window = relevant_data.iloc[i-n:i+1]
        
        # The first 'n' values are features, the last is the target
        x, y = window.iloc[:-1].values, window.iloc[-1]
        
        X.append(x)
        Y.append(y)
        dates.append(relevant_data.index[i]) # Store the date of the target 'y'

    # 4. Convert lists to NumPy arrays with correct shape and type
    dates = np.array(dates)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    # Reshape X for models like LSTMs that expect 3D input: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return dates, X, Y

dates, x, y = create_windowed_dataset(data, start_date_str="2020-01-01", end_date_str=now, n=3, feature_col='Close')


#Split the data into training, validation, and test sets
q80 = int(len(x)*0.88)
q90 = int(len(x)*0.93)

dates_train, x_train, y_train = dates[:q80], x[:q80], y[:q80]
dates_val, x_val, y_val = dates[q80:q90], x[q80:q90], y[q80:q90]
dates_test, x_test, y_test = dates[q90:], x[q90:], y[q90:]

#Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train.reshape(-1, 1)).reshape(x_train.shape)
x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(x_val.shape)
x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)

#Model training using LSTM
model = Sequential([layers.Input((3,1)), layers.LSTM(64), layers.Dense(32, activation = 'relu'), layers.Dense(32, activation = 'relu'), layers.Dense(1)])
model.compile(loss='mse', optimizer = Adam(learning_rate = 0.001), metrics = ['mean_absolute_error'])
model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 100)

train_pred = model.predict(x_train).flatten()
val_pred = model.predict(x_val).flatten()
test_pred = model.predict(x_test).flatten()

#Predict the next 14 days of stock prices
def predict_future_prices(model, last_window, num_predictions):
    predictions = []
    current_input = deepcopy(last_window)
    
    for _ in range(num_predictions):
        next_pred = model.predict(current_input.reshape(1, current_input.shape[0], 1)).flatten()
        predictions.append(next_pred[0])
        current_input = np.append(current_input[1:], next_pred).reshape(-1, 1)
    
    return predictions

# Predict the next 10 days of stock prices
last_window = x_test[-1] 
future_predictions = predict_future_prices(model, last_window, future_prediction_days)

# Print the future predicted prices
future_dates = pd.date_range(start=dates_test[-1], periods=future_prediction_days+1, freq='D')[1:]
future_dates = [date.strftime('%m-%d') for date in future_dates]
for i, pred in enumerate(future_predictions):
    print(f"Predicted price for {future_dates[i]}: {pred}")




# Data visualization
fig, axd = plt.subplot_mosaic([['upleft', 'right'], ['lowleft', 'right']], layout = 'constrained', figsize=(15, 8))

# Plot for training, validation, and test predictions
axd['upleft'].plot(dates_train, train_pred, c='red')
axd['upleft'].plot(dates_train, y_train, c='blue')
axd['upleft'].plot(dates_val, val_pred, c='green')
axd['upleft'].plot(dates_val, y_val, c='orange')
axd['upleft'].plot(dates_test, test_pred, c='purple')
axd['upleft'].plot(dates_test, y_test, c='black')
axd['upleft'].legend(['Train Prediction', 'Train', 'Validation Prediction', 'Validation', 'Test Prediction', 'Test'], loc = 'upper left')
axd['upleft'].set_ylabel('Stock Price')
axd['upleft'].set_xlabel('Date')
axd['upleft'].set_title('Stock Price Prediction for: ' + Symobl)

# Plot for future predictions
axd['lowleft'].plot(future_dates, future_predictions, c='teal', linestyle='-')
axd['lowleft'].set_ylabel('Stock Price for')
axd['lowleft'].set_xlabel('Date')
axd['lowleft'].set_title(Symobl + 'Stock Price Prediction for the Next 10 Days')
axd['lowleft'].tick_params(axis='x', labelsize=8, rotation=45)
axd['lowleft'].yaxis.set_major_formatter(mticker.ScalarFormatter())
axd['lowleft'].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Plot for total revenue over time
axd['right'].bar(datesIncStat, totalRevenue, color='maroon')
axd['right'].set_ylabel('Total Revenue')
axd['right'].set_xlabel('Date')
axd['right'].set_title('Total Revenue Over Time')
axd['right'].tick_params(axis='x', labelsize=8, rotation=45)
axd['right'].yaxis.set_major_formatter(mticker.ScalarFormatter())
axd['right'].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.show()



