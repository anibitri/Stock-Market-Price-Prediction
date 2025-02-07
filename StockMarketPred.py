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
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow, QLineEdit, QSpinBox, QCheckBox, QGroupBox, QFormLayout
from PyQt5.QtCore import QSize, Qt
import sys
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# Get the previous business day
def get_previous_business_day():
    today = datetime.datetime.now().date()
    one_day = timedelta(days=1)
    previous_day = today - one_day
    while previous_day.weekday() >= 5: 
        previous_day -= one_day
    return previous_day.strftime("%Y-%m-%d")

# Data preprocessing functions
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

def window_data_to_xy(window_df):
    data_as_np = window_df.to_numpy()
    dates = data_as_np[:, 0]
    middle = data_as_np[:, 1:-1]
    x = middle.reshape(len(dates), middle.shape[1], 1)
    y = data_as_np[:, -1]
    return dates, x.astype(np.float32), y.astype(np.float32)

# Predict the next 10 days of stock prices
def predict_future_prices(model, last_window, num_predictions):
    predictions = []
    current_input = deepcopy(last_window)
    
    for _ in range(num_predictions):
        next_pred = model.predict(current_input.reshape(1, current_input.shape[0], 1)).flatten()
        predictions.append(next_pred[0])
        current_input = np.append(current_input[1:], next_pred).reshape(-1, 1)
    
    return predictions

def predict_future_business_days(future_predictions, future_prediction_days, dates_test):
    # Generate a range of future business dates
    future_business_dates = pd.bdate_range(start=dates_test[-1], periods=future_prediction_days+1, freq='B')[1:]
    future_business_dates = [date.strftime('%m-%d') for date in future_business_dates]
    
    # Print the future predicted prices
    for i, pred in enumerate(future_predictions):
        print(f"Predicted price for {future_business_dates[i]}: {pred}")
    
    return future_business_dates

# Data visualization in PyQt window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Stock Market Price Prediction')
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Input section
        input_group = QGroupBox("Input Stock Symbol")
        input_layout = QFormLayout()
        self.stock_input = QLineEdit()
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        input_layout.addRow(QLabel("Stock Symbol:"), self.stock_input)
        input_layout.addRow(self.submit_button)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Create a tab widget
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs for each plot
        self.train_val_test_tab = QtWidgets.QWidget()
        self.future_predictions_tab = QtWidgets.QWidget()
        self.total_revenue_tab = QtWidgets.QWidget()
        
        self.tabs.addTab(self.train_val_test_tab, "Train/Val/Test Predictions")
        self.tabs.addTab(self.future_predictions_tab, "Future Predictions")
        self.tabs.addTab(self.total_revenue_tab, "Total Revenue")
        
        # Layouts for each tab
        self.train_val_test_layout = QtWidgets.QVBoxLayout(self.train_val_test_tab)
        self.future_predictions_layout = QtWidgets.QVBoxLayout(self.future_predictions_tab)
        self.total_revenue_layout = QtWidgets.QVBoxLayout(self.total_revenue_tab)
        
    def on_submit(self):
        Symobl = self.stock_input.text()
        symbolTicker = yf.Ticker(Symobl)
        incStat = symbolTicker.income_stmt.transpose()
        totalRevenue = incStat['Total Revenue']
        datesIncStat = totalRevenue.index
        datesIncStat = [str(date.date()) for date in datesIncStat]
        data = yf.download(Symobl, period="max")
        
        # Data preprocessing and model training
        prediction_days = 100
        future_prediction_days = 10 
        now = datetime.datetime.now().date().strftime('%Y-%m-%d')
        scaled = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaled.fit_transform(data['Close'].values.reshape(-1, 1))

        y = data['Close']
        x = data.drop(['Close'], axis=1)
        
        window_df = data_to_window_data(data, '2023-01-01', get_previous_business_day(), n=3)
        dates, x, y = window_data_to_xy(window_df)

        # Split the data into training, validation, and test sets
        q80 = int(len(x)*0.90)
        q90 = int(len(x)*0.95)

        dates_train, x_train, y_train = dates[:q80], x[:q80], y[:q80]
        dates_val, x_val, y_val = dates[q80:q90], x[q80:q90], y[q80:q90]
        dates_test, x_test, y_test = dates[q90:], x[q90:], y[q90:]

        # Model training using LSTM
        model = Sequential([layers.Input((3,1)), layers.LSTM(64), layers.Dense(32, activation = 'relu'), layers.Dense(32, activation = 'relu'), layers.Dense(1)])
        model.compile(loss='mse', optimizer = Adam(learning_rate = 0.001), metrics = ['mean_absolute_error'])
        model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 150)

        train_pred = model.predict(x_train).flatten()
        val_pred = model.predict(x_val).flatten()
        test_pred = model.predict(x_test).flatten()

        # Predict the next 10 days of stock prices
        last_window = x_test[-1] 
        future_predictions = predict_future_prices(model, last_window, future_prediction_days)
        future_prices = predict_future_business_days(future_predictions, future_prediction_days, dates_test)

        # Plot for training, validation, and test predictions
        train_val_test_fig, train_val_test_ax = plt.subplots(figsize=(10, 6))
        train_val_test_ax.plot(dates_train, train_pred, c='red')
        train_val_test_ax.plot(dates_train, y_train, c='blue')
        train_val_test_ax.plot(dates_val, val_pred, c='green')
        train_val_test_ax.plot(dates_val, y_val, c='orange')
        train_val_test_ax.plot(dates_test, test_pred, c='purple')
        train_val_test_ax.plot(dates_test, y_test, c='black')
        train_val_test_ax.legend(['Train Prediction', 'Train', 'Validation Prediction', 'Validation', 'Test Prediction', 'Test'], loc='upper left')
        train_val_test_ax.set_ylabel('Stock Price')
        train_val_test_ax.set_xlabel('Date')
        train_val_test_ax.set_title('Stock Price Prediction for: ' + Symobl)
        
        # Plot for future predictions
        future_predictions_fig, future_predictions_ax = plt.subplots(figsize=(10, 6))
        future_predictions_ax.plot(future_prices, future_predictions, c='teal', linestyle='-', marker='o')
        future_predictions_ax.set_ylabel('Stock Price')
        future_predictions_ax.set_xlabel('Date')
        future_predictions_ax.set_title(Symobl + ' Stock Price Prediction for the Next 10 Days')
        future_predictions_ax.tick_params(axis='x', labelsize=8, rotation=45)
        future_predictions_ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        future_predictions_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Plot for total revenue over time
        total_revenue_fig, total_revenue_ax = plt.subplots(figsize=(10, 6))
        total_revenue_ax.bar(datesIncStat, totalRevenue, color='maroon')
        total_revenue_ax.set_ylabel('Total Revenue')
        total_revenue_ax.set_xlabel('Date')
        total_revenue_ax.set_title('Total Revenue Over Time')
        total_revenue_ax.tick_params(axis='x', labelsize=8, rotation=45)
        total_revenue_ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        total_revenue_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Add matplotlib figures to the tabs
        train_val_test_canvas = FigureCanvasQTAgg(train_val_test_fig)
        future_predictions_canvas = FigureCanvasQTAgg(future_predictions_fig)
        total_revenue_canvas = FigureCanvasQTAgg(total_revenue_fig)
        
        self.train_val_test_layout.addWidget(train_val_test_canvas)
        self.future_predictions_layout.addWidget(future_predictions_canvas)
        self.total_revenue_layout.addWidget(total_revenue_canvas)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
