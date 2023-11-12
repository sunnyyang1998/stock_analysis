from json import JSONDecodeError
import os
from httpx import HTTPError
import requests
import pandas as pd

# 国家及其ISO代码列表
countries = {
    'China': 'CN',
    'Japan': 'JP',
    'South Korea': 'KR',
    'India': 'IN',
    'Indonesia': 'ID',
    'Malaysia': 'MY',
    'Vietnam': 'VN',
    'Thailand': 'TH',
    'Singapore': 'SG',
    'Philippines': 'PH',
    'United Kingdom': 'GB',
    'United States': 'US'
}

# 经济指标及其对应的代码
indicators = {
    'Interest Rate': 'FR.INR.RINR',
    'GDP Growth Rate': 'NY.GDP.MKTP.KD.ZG',
    'Inflation Rate': 'FP.CPI.TOTL.ZG',
    'Unemployment Rate': 'SL.UEM.TOTL.ZS',
    'Industrial Production Growth Rate': 'NV.IND.MANF.KD.ZG',
    'Trade Balance': 'NE.RSB.GNFS.CD',
    'Current Account Balance': 'BN.CAB.XOKA.CD',
    'External Debt': 'DT.DOD.DECT.CD',
    'Foreign Direct Investment, net inflows': 'BX.KLT.DINV.WD.GD.ZS',
    'Exchange Rate against USD': 'PA.NUS.FCRF',
    'Government Debt to GDP': 'GC.DOD.TOTL.GD.ZS',
    'Foreign Reserves': 'FI.RES.TOTL.CD',
    'GDP per capita': 'NY.GDP.PCAP.CD',
    'GDP': 'NY.GDP.MKTP.CD',
    'Population': 'SP.POP.TOTL',
    'Stock Market Capitalization to GDP': 'CM.MKT.LCAP.GD.ZS'
}

START_DATE = "1973"
END_DATE = "2023"

def fetch_indicator_data(country_code, indicator_code):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={START_DATE}:{END_DATE}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 抛出 HTTP 错误
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            return []

        return data[1]
    except HTTPError as e:
        print(f"HTTPError: 请求失败 - {e}")
        return []
    except JSONDecodeError as e:
        print(f"JSONDecodeError: 响应不是有效的 JSON 格式 - {e}")
        return []

parent_dir = "countries"
if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)

for country, code in countries.items():
    print(f"Fetching data for {country}...")
    
    country_data = {}
    for indicator_name, indicator_code in indicators.items():
        indicator_values = fetch_indicator_data(code, indicator_code)
        for entry in indicator_values:
            year = entry['date']
            value = entry['value']
            if year not in country_data:
                country_data[year] = {}
            country_data[year][indicator_name] = value

    df = pd.DataFrame(country_data).transpose()
    
    country_dir = os.path.join(parent_dir, code)
    if not os.path.exists(country_dir):
        os.mkdir(country_dir)
    csv_file = os.path.join(country_dir, f"{code}.csv")
    df.to_csv(csv_file)

print("All data fetched and saved!")
