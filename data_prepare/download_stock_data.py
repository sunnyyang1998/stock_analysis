from tqdm import tqdm
import yfinance as yf
import os

def fetch_stock_data(ticker, progress_bar):  # 添加了progress_bar参数
    try:
        stock_data = yf.download(ticker, period="max", progress=False)  # 确保yfinance不显示其自己的进度条
        if stock_data.empty:
            return None
        return stock_data
    except Exception as e:
        progress_bar.write(f"Failed download for {ticker}: {str(e)}")  # 输出错误信息到进度条下一行
        return None

def fetch_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        if any([df.empty for df in [cash_flow, balance_sheet, income_statement]]):  # 检查任何数据帧是否为空
            # print(f"Financial data for {ticker} is empty.")
            return None, None, None
        return cash_flow, balance_sheet, income_statement
    except Exception as e:
        # print(f"Error fetching financial data for {ticker}: {e}")
        return None, None, None

# 从txt文件中读取股票代码
with open("international_companies.txt", "r") as f:
    international_companies = f.readlines()
    
with open("local_companies.txt", "r") as f:
    local_companies = f.readlines()

# 合并两个列表
all_companies = local_companies + international_companies

# 创建主文件夹 'data'
data_dir = "data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# 使用tqdm来包装all_companies以显示进度条
with tqdm(all_companies, desc="Fetching data", total=len(all_companies)) as progress_bar:
    for company in progress_bar:
        company = company.strip()  # 去除尾部的换行符

        # 获取并保存数据
        stock_data = fetch_stock_data(company, progress_bar)
        if stock_data is not None:
            # 为每家公司创建文件夹
            company_dir = os.path.join(data_dir, company)
            if not os.path.exists(company_dir):
                os.mkdir(company_dir)
            
            stock_dir = os.path.join(company_dir, 'stock')
            if not os.path.exists(stock_dir):
                os.mkdir(stock_dir)
            stock_data.to_csv(os.path.join(stock_dir, f"{company}_stock_data.csv"))

            cash_flow, balance_sheet, income_statement = fetch_financial_data(company)
            if all([df is not None for df in [cash_flow, balance_sheet, income_statement]]):
                finance_dir = os.path.join(company_dir, 'finance_data')
                if not os.path.exists(finance_dir):
                    os.mkdir(finance_dir)
                cash_flow.to_csv(os.path.join(finance_dir, f"{company}_cash_flow.csv"))
                balance_sheet.to_csv(os.path.join(finance_dir, f"{company}_balance_sheet.csv"))
                income_statement.to_csv(os.path.join(finance_dir, f"{company}_income_statement.csv"))

print("All data fetched and saved!")