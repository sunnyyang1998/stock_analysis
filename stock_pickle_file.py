import os
import pandas as pd
import pickle
import numpy as np

data_dir = "data"

def preprocess_stock_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date']).dt.year
    data = data.dropna()

    stock_features = ['High', 'Low', 'Close', 'Adj Close', 'Volume', 'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j', 'roc', 'maroc']
    stock_labels = ['Open', 'change', 'signal']
    X_stock = data[stock_features].values
    y_stock = data[stock_labels].values

    return X_stock, y_stock

def preprocess_finance_and_macro_data(finance_path, macro_path):
    finance_df = pd.read_csv(finance_path)
    macro_df = pd.read_csv(macro_path)

    finance_df['Date'] = pd.to_datetime(finance_df['Date']).dt.year
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.year

    merged_df = pd.merge(finance_df, macro_df, on='Date', how='outer')
    merged_df.fillna(0, inplace=True)

    return merged_df.values

for company_folder in os.listdir(data_dir):
    if company_folder.startswith('.'):
        continue

    company_yearly_data = {}

    stock_dir = os.path.join(data_dir, company_folder, 'stock')
    if os.path.isdir(stock_dir):
        for file in os.listdir(stock_dir):
            if file.endswith('.csv') and not file.startswith('.'):
                stock_data_path = os.path.join(stock_dir, file)
                try:
                    X_stock, y_stock = preprocess_stock_data(stock_data_path)
                    year = file.split('_')[0]
                    company_yearly_data[year] = {'X_stock': X_stock, 'y_stock': y_stock}
                except Exception as e:
                    print(f"Error in stock data: {e}")

    finance_data_dir = os.path.join(data_dir, company_folder, 'processed_finance_data')
    macro_data_dir = os.path.join(data_dir, company_folder, 'processed_macro_data')
    if os.path.isdir(finance_data_dir) and os.path.isdir(macro_data_dir):
        for finance_file in os.listdir(finance_data_dir):
            if finance_file.endswith('.csv') and not finance_file.startswith('.'):
                finance_data_path = os.path.join(finance_data_dir, finance_file)
                macro_data_path = os.path.join(macro_data_dir, finance_file)
                
                if os.path.exists(macro_data_path):
                    try:
                        finance_macro_data = preprocess_finance_and_macro_data(finance_data_path, macro_data_path)
                        year = finance_file.split('_')[0]
                        if year in company_yearly_data:
                            company_yearly_data[year]['finance_macro'] = finance_macro_data
                    except Exception as e:
                        print(f"Error in finance/macro data: {e}")

    pickle_dir = os.path.join(data_dir, company_folder, 'pickle_file')
    os.makedirs(pickle_dir, exist_ok=True)

    for year, data_dict in company_yearly_data.items():
        X_stock = data_dict.get('X_stock', np.array([]))
        y_stock = data_dict.get('y_stock', np.array([]))
        finance_macro = data_dict.get('finance_macro', np.array([]))

        combined_data = {
            'X_stock': X_stock, 
            'y_stock': y_stock, 
            'finance_macro': finance_macro
        }

        pickle_file_path = os.path.join(pickle_dir, f'{company_folder}_feature.pkl')
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"Data for {year} saved to {pickle_file_path}")
