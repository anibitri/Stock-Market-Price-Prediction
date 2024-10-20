import pandas as pd
import yfinance as yf
import datetime
from datetime import timedelta
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
Symobl = input("Enter the stock symbol: ")
symbolTicker = yf.Ticker(Symobl)
incStat = symbolTicker.income_stmt.transpose()
totalRevenue = incStat['Total Revenue']
datesIncStat = totalRevenue.index
datesIncStat = [str(date.date()) for date in datesIncStat]
data = yf.download(Symobl, period="max")

#Global Variables
prediction_days = 100
future_prediction_days = 10 
now = datetime.datetime.now().date().strftime('%Y-%m-%d')
scaled = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaled.fit_transform(data['Close'].values.reshape(-1, 1))

#Get the previous business day
def get_previous_business_day():
    today = datetime.datetime.now().date()
    one_day = timedelta(days=1)
    previous_day = today - one_day
    while previous_day.weekday() >= 5: 
        previous_day -= one_day
    return previous_day.strftime("%Y-%m-%d")

#Data Preprocessing
y = data['Close']
x = data.drop(['Close'], axis=1)
def str_to_dt(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

def data_to_window_data(dataframe, firstdate, lastdate, n=3):
    first_date = str_to_dt(firstdate)
    last_date = str_to_dt(lastdate)

    target_date = first_date
    
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
        
        target_date = next_date

        if target_date == last_date:
            last_time = True
    
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

window_df = data_to_window_data(data, '2023-01-01', get_previous_business_day(), n=3)


def window_data_to_xy(window_df):
    data_as_np = window_df.to_numpy()
    dates = data_as_np[:, 0]
    middle = data_as_np[:, 1:-1]
    x = middle.reshape(len(dates), middle.shape[1], 1)
    y = data_as_np[:, -1]
    return dates, x.astype(np.float32), y.astype(np.float32)

dates, x, y = window_data_to_xy(window_df)


#Split the data into training, validation, and test sets
q80 = int(len(x)*0.85)
q90 = int(len(x)*0.92)

dates_train, x_train, y_train = dates[:q80], x[:q80], y[:q80]
dates_val, x_val, y_val = dates[q80:q90], x[q80:q90], y[q80:q90]
dates_test, x_test, y_test = dates[q90:], x[q90:], y[q90:]

#Model training using LSTM
model = Sequential([layers.Input((3,1)), layers.LSTM(64), layers.Dense(32, activation = 'relu'), layers.Dense(32, activation = 'relu'), layers.Dense(1)])
model.compile(loss='mse', optimizer = Adam(learning_rate = 0.001), metrics = ['mean_absolute_error'])
model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 150)

train_pred = model.predict(x_train).flatten()
val_pred = model.predict(x_val).flatten()
test_pred = model.predict(x_test).flatten()

#Predict the next 10 days of stock prices
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

def predict_future_business_days(future_predictions, future_prediction_days, dates_test):
    # Generate a range of future business dates
    future_business_dates = pd.bdate_range(start=dates_test[-1], periods=future_prediction_days+1, freq='B')[1:]
    future_business_dates = [date.strftime('%m-%d') for date in future_business_dates]
    
    # Print the future predicted prices
    for i, pred in enumerate(future_predictions):
        print(f"Predicted price for {future_business_dates[i]}: {pred}")
    
    return future_business_dates

future_prices = predict_future_business_days(future_predictions, future_prediction_days, dates_test)




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
axd['lowleft'].plot(future_prices, future_predictions, c='teal', linestyle='-', marker='o')
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