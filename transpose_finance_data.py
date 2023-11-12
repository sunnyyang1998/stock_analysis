import os
import pandas as pd

from fix_finance_data import modify_all_csv_headers

def transpose_all_csv_in_dir(data_dir="data"):
    def transpose_csv(file_path):
        # 读取CSV文件
        df = pd.read_csv(file_path, index_col=0)

        # 转置数据框
        df_transposed = df.transpose()

        # 重新设置列名为 'Date'
        df_transposed.columns.name = 'Date'

        # 保存转置后的数据框回到相同的CSV文件
        df_transposed.to_csv(file_path)

    # 遍历"data"目录下的所有公司文件夹
    for company in os.listdir(data_dir):
        company_dir = os.path.join(data_dir, company)

        # 检查这是否是一个目录（而不是文件）
        if os.path.isdir(company_dir):
            finance_dir = os.path.join(company_dir, 'finance_data')

            # 检查 finance_data 文件夹是否存在
            if os.path.exists(finance_dir):
                for finance_file in os.listdir(finance_dir):
                    if finance_file.endswith('.csv'):
                        transpose_csv(os.path.join(finance_dir, finance_file))

    print("All CSV files in finance_data folders have been transposed!")

# 使用函数
# 使用函数
modify_all_csv_headers()
transpose_all_csv_in_dir()
modify_all_csv_headers()
