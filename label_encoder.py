import os
import pandas as pd

data_dir = "data"

# 遍历"data"目录下的所有项目
for company_folder in os.listdir(data_dir):
    # 忽略隐藏文件和目录
    if company_folder.startswith('.'):
        continue
    
    # 构建finance_data子文件夹路径和macro_data子文件夹路径
    finance_data_dir = os.path.join(data_dir, company_folder, 'processed_finance_data')
    macro_data_dir = os.path.join(data_dir, company_folder, 'processed_macro_data')
    
    # 检查子文件夹是否存在且为目录
    if os.path.isdir(finance_data_dir) and os.path.isdir(macro_data_dir):
        # 遍历finance_data和macro_data子文件夹中的所有文件
        for finance_file in os.listdir(finance_data_dir):
            if finance_file.endswith('.csv') and not finance_file.startswith('.'):
                finance_data_path = os.path.join(finance_data_dir, finance_file)
                
                # 找到对应的宏观数据文件
                macro_data_path = os.path.join(macro_data_dir, finance_file)
                
                if os.path.exists(macro_data_path):
                    try:
                        # 读取财务数据和宏观数据
                        finance_df = pd.read_csv(finance_data_path)
                        macro_df = pd.read_csv(macro_data_path)

                        # 合并数据集
                        merged_df = pd.merge(finance_df, macro_df, on='Date', how='outer')

                        # 处理额外的列（可选）
                        # 例如，使用0填充缺失值
                        merged_df.fillna(0, inplace=True)

                        # 保存合并后的数据
                        merged_data_path = os.path.join(data_dir, company_folder, f'merged_data_{finance_file}')
                        merged_df.to_csv(merged_data_path, index=False)

                        print(f"Merged data saved to {merged_data_path}")
                    except Exception as e:
                        print(f"An error occurred while merging data {finance_file} for {company_folder}: {e}")
