# 函数：检查Pickle文件内容
import pickle


def check_pickle_content(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return list(data.keys())  # 返回字典中所有键的列表

# 示例用法
pickle_file_path = '/Users/qiaoxi/Desktop/设计/stock_analysis/data/000001.SZ/pickle_file/000001.SZ_feature.pkl'  # 替换为实际文件路径
print(check_pickle_content(pickle_file_path))