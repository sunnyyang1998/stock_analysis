import os
import shutil

data_dir = "data"

# 遍历"data"目录下的所有可能的公司文件夹
for company_folder in os.listdir(data_dir):
    company_path = os.path.join(data_dir, company_folder)
    
    # 检查是否为目录以及所需的子目录是否存在
    if os.path.isdir(company_path):
        required_subdirs = ['finance_data', 'stock', 'processed_finance_data']
        subdirs_exist = all(os.path.isdir(os.path.join(company_path, subdir)) for subdir in required_subdirs)
        
        if not subdirs_exist:
            # 如果缺少任何子文件夹，删除整个{company_folder}
            shutil.rmtree(company_path)
            print(f"Deleted company folder due to missing subdirectories: {company_folder}")
        else:
            print(f"All required subdirectories exist for company: {company_folder}")

print(f"Finished checking and cleaning the data directory.")
