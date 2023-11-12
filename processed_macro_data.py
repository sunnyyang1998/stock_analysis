import os
import pandas as pd

# 假设的根目录
root_dir = "data"
# 源宏观数据目录
source_macro_dir = "countries"
# 目标宏观数据目录名
target_macro_dir_name = "processed_macro_data"

# 遍历每个公司文件夹
for company_folder in os.listdir(root_dir):
    company_dir = os.path.join(root_dir, company_folder)
    target_macro_dir = os.path.join(company_dir, target_macro_dir_name)

    # 检查是否为目录并且不是系统文件夹，如.DS_Store
    if os.path.isdir(company_dir) and not company_folder.startswith('.'):
        # 如果目标目录不存在，则创建它
        os.makedirs(target_macro_dir, exist_ok=True)

        # 初始化一个空的DataFrame来存储合并后的数据
        combined_macro_data = pd.DataFrame()

        # 遍历源宏观数据目录下的每个国家文件夹，合并数据
        for country_folder in os.listdir(source_macro_dir):
            country_path = os.path.join(source_macro_dir, country_folder)
            
            # 确保是一个文件夹
            if os.path.isdir(country_path):
                for file_name in os.listdir(country_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(country_path, file_name)

                        # 读取CSV文件
                        df = pd.read_csv(file_path)
                        
                        # 将'Interest Rate'等指标名称作为一列数据，并将年份作为行索引
                        df = df.set_index('Date').transpose()
                        
                        # 为列添加国家代码作为前缀，以避免合并时的冲突
                        df = df.add_prefix(country_folder + '_')

                        # 将年份索引转换为列
                        df.reset_index(inplace=True)
                        df.rename(columns={'index': 'Date'}, inplace=True)

                        # 合并数据
                        if combined_macro_data.empty:
                            combined_macro_data = df
                        else:
                            combined_macro_data = pd.merge(combined_macro_data, df, on='Date', how='outer')

        # 将NaN值填充为0
        combined_macro_data.fillna(0, inplace=True)

        # 将合并后的宏观数据保存在目标目录中
        output_file_path = os.path.join(target_macro_dir, f"{company_folder}_processed_macro_data.csv")
        combined_macro_data.to_csv(output_file_path, index=False)

        print(f"Processed macro data for {company_folder} saved to {output_file_path}")
