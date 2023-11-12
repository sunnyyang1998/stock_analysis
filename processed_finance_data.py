import os
import pandas as pd

data_dir = "data"

# 遍历"data"目录下的所有公司文件夹
for company_folder in os.listdir(data_dir):
    company_path = os.path.join(data_dir, company_folder)
    # 检查路径是否为目录
    if os.path.isdir(company_path):
        finance_dir = os.path.join(company_path, 'finance_data')
        processed_finance_dir = os.path.join(company_path, 'processed_finance_data')
        
        # 检查finance_data文件夹是否存在
        if not os.path.exists(finance_dir):
            print(f"finance_data directory does not exist for company {company_folder}. Skipping.")
            continue
        
        # 确保processed_finance_data文件夹存在
        if not os.path.exists(processed_finance_dir):
            os.makedirs(processed_finance_dir)
        
        combined_df = pd.DataFrame()
        
        # 遍历finance_data文件夹中的所有CSV文件
        for csv_file in os.listdir(finance_dir):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(finance_dir, csv_file)
                df = pd.read_csv(csv_path)
                
                # 提取年份作为索引
                df['Date'] = pd.to_datetime(df['Date']).dt.year
                df.set_index('Date', inplace=True)
                
                # 将每个文件的数据加入到总的DataFrame中
                combined_df = pd.concat([combined_df, df], axis=1)
        
        # 处理空值，您可以根据需要修改这里的处理方式
        combined_df.fillna(0, inplace=True)

        # 保存合并后的DataFrame到processed_finance_data文件夹
        combined_filename = f"{company_folder}_combined.csv"
        combined_path = os.path.join(processed_finance_dir, combined_filename)
        combined_df.to_csv(combined_path)

print(f"All CSV files in finance_data folders have been processed and saved in the processed_finance_data folders.")
