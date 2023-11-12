import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

# 读取数据
data = pd.read_csv('002773.SZ.csv')
data['Date'] = pd.to_datetime(data['Date'])

# 获取唯一的年份列表
years = data['Date'].dt.year.unique()

# 存储每年的性能指标
performance_by_year = {}

for year in years:
    year_data = data[data['Date'].dt.year == year].copy()
    year_data['MA5'] = year_data['Close'].rolling(window=5).mean()
    year_data['MA10'] = year_data['Close'].rolling(window=10).mean()
    year_data['MA20'] = year_data['Close'].rolling(window=20).mean()
    year_data['ROC'] = ((year_data['Close'] - year_data['Close'].shift(12)) / year_data['Close'].shift(12)) * 100
    year_data['MAROC'] = year_data['ROC'].rolling(window=5).mean()
    
    features = ['MA5', 'MA10', 'MA20', 'ROC', 'MAROC']
    label = 'Close'
    year_data = year_data.dropna()
    X = year_data[features].values
    y = year_data[label].values
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    # 使用CNN模型
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance_by_year[year] = {'MSE': mse, 'R2': r2}

    print(f'Year {year}:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R2): {r2}')
    print()

model.save('002773.SZ_CNN.h5')

future_predictions_close = []
current_X = X_test[-1].copy()

num_days_to_predict = 1
for _ in range(num_days_to_predict):
    next_day_prediction = model.predict(np.reshape(current_X, (1, current_X.shape[0], 1)))
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    future_predictions_close.append(next_day_prediction[0][0])
    current_X = np.roll(current_X, -1)
    current_X[-1] = next_day_prediction[0][0]

print(f'Future Predictions (Close): {future_predictions_close}')
