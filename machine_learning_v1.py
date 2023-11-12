import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*frozen modules.*")

data_dir = "data"  # 存放公司数据的目录

# 函数：读取和预处理数据
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])

    # 选择特征和标签
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j', 'change', 'roc', 'maroc']
    label = 'Close'  # 或者选择其他您想预测的标签
    data = data.dropna()
    X = data[features].values
    y = data[label].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    return X, y, scaler

# 函数：构建CNN模型
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 初始化模型
model = None

# 获取总文件数以用于进度条
total_files = 0
for company_folder in os.listdir(data_dir):
    company_path = os.path.join(data_dir, company_folder)
    
    # 确保是目录
    if not os.path.isdir(company_path):
        continue

    stock_dir = os.path.join(company_path, 'stock')
    total_files += len([f for f in os.listdir(stock_dir) if f.endswith('.csv')])

# 创建整个训练过程的进度条
with tqdm(total=total_files, desc='Training Progress') as pbar:
    # 遍历公司文件夹
    for company_folder in os.listdir(data_dir):
        company_path = os.path.join(data_dir, company_folder)
        
        # 确保是目录
        if not os.path.isdir(company_path):
            continue

        stock_dir = os.path.join(company_path, 'stock')
        
        # 遍历股票数据文件
        for stock_file in os.listdir(stock_dir):
            file_path = os.path.join(stock_dir, stock_file)

            # 检查是否为CSV文件
            if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
                continue

            X, y, scaler = preprocess_data(file_path)

            # 分割数据集
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # 调整数据形状以适应CNN
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # 对第一家公司的第一个文件构建模型，对其余公司调整模型
            if model is None:
                model = build_cnn_model((X_train.shape[1], 1))

            # 训练或调整模型
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # 评估模型
            y_pred = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # print(f'{stock_file} - Mean Squared Error (MSE): {mse}')
            # print(f'{stock_file} - R-squared (R2): {r2}')
            # print()

            # 更新进度条
            pbar.update(1)

# 最终模型评估
final_mse = 0.0
final_r2 = 0.0
total_files = 0

# 重新遍历以计算总体指标
for company_folder in os.listdir(data_dir):
    company_path = os.path.join(data_dir, company_folder)
    
    # 确保是目录
    if not os.path.isdir(company_path):
        continue

    stock_dir = os.path.join(company_path, 'stock')
    
    # 遍历股票数据文件
    for stock_file in os.listdir(stock_dir):
        file_path = os.path.join(stock_dir, stock_file)

        # 检查是否为CSV文件
        if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
            continue

        X, y, scaler = preprocess_data(file_path)

        # 分割数据集
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 调整数据形状以适应CNN
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # 评估模型
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        final_mse += mse
        final_r2 += r2
        total_files += 1

# 计算平均指标
final_mse /= total_files
final_r2 /= total_files

print(f'Final Mean Squared Error (MSE): {final_mse}')
print(f'Final R-squared (R2): {final_r2}')

# 保存模型
model.save('final_CNN_model.h5')
