import os

def modify_all_csv_headers(data_dir="data"):
    def modify_csv_header(file_path):
        # 读取文件
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 修改第一行
        if lines[0].startswith(","):
            lines[0] = "Date" + lines[0]

        # 保存修改
        with open(file_path, 'w') as f:
            f.writelines(lines)

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
                        modify_csv_header(os.path.join(finance_dir, finance_file))

    print("Headers in all CSV files have been modified!")

