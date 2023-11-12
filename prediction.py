import pickle
import numpy as np
from keras.models import load_model

# 加载模型
model = load_model('final_CNN_model.h5')

# 加载和准备数据
# 假设你有一个新的pickle文件用于预测
with open('new_data.pkl', 'rb') as f:
    new_data = pickle.load(f)
X_new = new_data['X_stock']
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

# 使用模型进行预测
predictions = model.predict(X_new)

# 处理/展示预测结果
# 这取决于你的具体需求，例如：
print("预测结果:", predictions.flatten())
