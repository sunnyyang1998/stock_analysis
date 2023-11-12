import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from tqdm import tqdm
import concurrent.futures

# 解决调试器警告
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# 数据目录
data_dir = "data"

# 函数：从Pickle文件加载数据
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_stock'], data['y_stock'], data['finance_macro']

# 函数：构建CNN模型
def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=30, activation='relu', kernel_regularizer=l2(0.001))) # 另一个全连接层
    model.add(Dense(units=1))
    model.compile(optimizer='adamax', loss='mean_squared_error')
    return model

# 初始化模型
model = None

# 使用多线程加载pickle数据
all_data = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for company_folder in os.listdir(data_dir):
        company_path = os.path.join(data_dir, company_folder)
        
        if os.path.isdir(company_path):
            pickle_file_path = os.path.join(company_path, 'pickle_file', f'{company_folder}_feature.pkl')
            if os.path.isfile(pickle_file_path):
                futures.append(executor.submit(load_data_from_pickle, pickle_file_path))

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Loading Data'):
        X_stock, y_stock, finance_macro = future.result()  # Updated this line
        if X_stock is not None and y_stock is not None:
            X_stock = np.reshape(X_stock, (X_stock.shape[0], X_stock.shape[1], 1))
            all_data.append((X_stock, y_stock))  # Assuming you only need X_stock and y_stock

# 数据分割和模型训练
for X, y in tqdm(all_data, desc='Preparing and Training Model'):
    if model is None:
        input_shape = X.shape[1:]
        model = build_cnn_model(input_shape)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 评估模型
final_mse = 0.0
total_files = len(all_data)

for X, y in all_data:
    mse = model.evaluate(X_test, y_test, verbose=0)
    final_mse += mse

final_mse /= total_files
print(f'Final Mean Squared Error (MSE): {final_mse}')

# 保存模型
model.save('final_CNN_model.h5')
