import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping

# 设置目录路径
root_dir = '/Users/qiaoxi/Desktop/设计/stock_analysis'  # 替换为您的项目根目录路径
data_dir = os.path.join(root_dir, 'data')
models_dir = os.path.join(root_dir, 'models')
predictions_dir = os.path.join(root_dir, 'predictions')

# 确保模型和预测目录存在
os.makedirs(models_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# 定义一个函数来处理数据和训练模型
def process_and_train(company_data_path):
    data = pd.read_csv(company_data_path)
    # 数据预处理（您可能需要根据实际情况修改）
    data['Date'] = pd.to_datetime(data['Date']).dt.year
    data.set_index('Date', inplace=True)

    # 提取特征（您可能需要添加更多特征）
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['ROC'] = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12)) * 100
    data['MAROC'] = data['ROC'].rolling(window=5).mean()
    data.dropna(inplace=True)

    features = ['MA5', 'MA10', 'MA20', 'ROC', 'MAROC']
    X = data[features].values
    y = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # 创建模型
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])

    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
    r2 = r2_score(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
    
    return model, scaler, mse, r2

# 获取所有子目录列表
company_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
company_folders.sort()  # 确保有一致的顺序，例如按名称

# 使用第一家公司数据训练模型
first_company_folder = company_folders[0]
first_company_data_path = os.path.join(data_dir, first_company_folder, 'stock_data.csv')  # 文件名可能需要修改
model, scaler, mse, r2 = process_and_train(first_company_data_path)
print(f"First company {first_company_folder}: MSE={mse}, R^2={r2}")

# 保存第一家公司的模型
model.save(os.path.join(models_dir, 'first_company_model.h5'))

# 使用其他公司数据调整模型
for company_folder in company_folders[1:]:
    company_data_path = os.path.join(data_dir, company_folder, 'stock_data.csv')  # 文件名可能需要修改
    _, _, mse, r2 = process_and_train(company_data_path)
    print(f"Company {company_folder}: MSE={mse}, R^2={r2}")

print("Completed model training and evaluation for all companies.")
