# 示例股票代码列表
shenzhen_companies = [f"000{str(i).zfill(3)}.SZ" for i in range(1000)]
shanghai_companies = [f"600{str(i).zfill(3)}.SS" for i in range(1000)]

# 合并两个列表
all_companies = shenzhen_companies + shanghai_companies

# 将所有股票代码写入文件
with open('local_companies.txt', 'w') as file:
    for company in all_companies:
        file.write(f"{company}\n")

print("Stock codes saved to local_companies.txt!")

#### international companies

import requests
from bs4 import BeautifulSoup

def get_sp500_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    table = soup.find('table', {'class': 'wikitable sortable'})
    symbols = [row.findAll('td')[0].text.strip() for row in table.findAll('tr')[1:]]
    
    return symbols

# 获取S&P 500的公司股票代码
sp500_symbols = get_sp500_companies()

# 如果数量不足500家，你可以考虑从其他指数中获取更多公司，或者直接截取前500家
selected_symbols = sp500_symbols[:600]

# 将股票代码保存到文件
with open('international_companies.txt', 'w') as file:
    for symbol in selected_symbols:
        file.write(f"{symbol}\n")

print("International stock codes saved to international_companies.txt!")

