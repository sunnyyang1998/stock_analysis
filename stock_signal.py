import os
import pandas as pd
import talib
import numpy as np

data_dir = "data"  # 确保这是您存放CSV文件的实际目录路径

# 遍历"data"目录下的所有公司文件夹
for company_folder in os.listdir(data_dir):
    company_path = os.path.join(data_dir, company_folder)
    if os.path.isdir(company_path):
        stock_data_path = os.path.join(company_path, 'stock', f"{company_folder}_stock_data.csv")
        if os.path.exists(stock_data_path):
            # 读取股票数据
            stock_df = pd.read_csv(stock_data_path)

            # 转换日期格式
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])

            # 确保所有计算依赖的列都存在
            if 'Close' in stock_df.columns and 'High' in stock_df.columns and 'Low' in stock_df.columns:
                # 填充NaN值为0
                stock_df.fillna(0, inplace=True)

                # 计算技术指标
                stock_df['ma5'] = talib.MA(stock_df['Close'], timeperiod=5)
                stock_df['ma10'] = talib.MA(stock_df['Close'], timeperiod=10)
                stock_df['ma20'] = talib.MA(stock_df['Close'], timeperiod=20)
                upperband, middleband, lowerband = talib.BBANDS(stock_df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                macd, macdsignal, macdhist = talib.MACD(stock_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
                stock_df['rsi'] = talib.RSI(stock_df['Close'], timeperiod=14)
                slowk, slowd = talib.STOCH(stock_df['High'], stock_df['Low'], stock_df['Close'], fastk_period=9, slowk_period=3, slowd_period=3)
                stock_df['roc'] = talib.ROC(stock_df['Close'], timeperiod=10)
                stock_df['maroc'] = talib.MA(stock_df['roc'], timeperiod=10)
                
                # 将技术指标的结果添加到DataFrame
                stock_df['k'] = slowk
                stock_df['d'] = slowd
                stock_df['j'] = 3 * stock_df['k'] - 2 * stock_df['d']
                stock_df['upperband'] = upperband
                stock_df['middleband'] = middleband
                stock_df['lowerband'] = lowerband
                stock_df['macd'] = macd
                stock_df['macdsignal'] = macdsignal
                stock_df['macdhist'] = macdhist
                
                # 计算涨幅
                stock_df['change'] = stock_df['Close'].pct_change() * 100

                # 生成交易信号
                buy_signals = (
                    (stock_df['ma5'] > stock_df['ma10']) & 
                    (stock_df['ma10'] > stock_df['ma20']) & 
                    (stock_df['Close'] > stock_df['middleband']) & 
                    (stock_df['macd'] > 0) & 
                    (stock_df['rsi'] < 70) & 
                    (stock_df['k'] > stock_df['d']) &
                    (stock_df['roc'] > 0) &  # ROC指标为正
                    (stock_df['maroc'] > 0) &  # MAROC指标为正
                    (stock_df['change'] > 0.5)  # 当日涨幅超过0.5%
                )
                
                sell_signals = (
                    (stock_df['ma5'] < stock_df['ma10']) & 
                    (stock_df['ma10'] < stock_df['ma20']) & 
                    (stock_df['Close'] < stock_df['middleband']) & 
                    (stock_df['macd'] < 0) & 
                    (stock_df['rsi'] > 30) & 
                    (stock_df['k'] < stock_df['d']) &
                    (stock_df['roc'] < 0) &  # ROC指标为负
                    (stock_df['maroc'] < 0) &  # MAROC指标为负
                    (stock_df['change'] < -0.5)  # 当日跌幅低于-0.5%
                )

                # 初始化信号为0 (持平)
                stock_df['signal'] = 0
                stock_df.loc[buy_signals, 'signal'] = 1   # 买入信号
                stock_df.loc[sell_signals, 'signal'] = -1  # 卖出信号

                # 对信号列进行热编码
                signal_dummies = pd.get_dummies(stock_df['signal'], prefix='signal')
                stock_df = pd.concat([stock_df, signal_dummies], axis=1)

                # 选择需要的列，包括热编码的列
                columns = [
                    'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                    'ma5', 'ma10', 'ma20', 'upperband', 'middleband', 'lowerband', 
                    'macd', 'macdsignal', 'macdhist', 'rsi', 'k', 'd', 'j', 'change', 
                    'roc', 'maroc', 'signal', 'signal_-1', 'signal_0', 'signal_1'
                ]
                stock_df = stock_df[columns]

                # 保存更新后的DataFrame到CSV文件
                stock_df.to_csv(stock_data_path, index=False)

                print(f"Updated stock data with one-hot encoded signals for {company_folder} has been saved to {stock_data_path}")
            else:
                print(f"Required columns are missing in the file {stock_data_path}.")
        else:
            print(f"Stock data file does not exist for company {company_folder}. Skipping.")
