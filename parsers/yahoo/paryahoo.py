import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_daily_data(ticker_symbol, period='1y'):
    ticker = yf.Ticker(ticker_symbol)

    data = ticker.history(period=period, interval='1d')
    
    data = data.reset_index()
    
    daily_data = pd.DataFrame({
        'Дата': data['Date'].dt.strftime('%Y-%m-%d'),
        'Цена открытия': data['Open'].round(2),
        'Максимум': data['High'].round(2),
        'Минимум': data['Low'].round(2),
        'Цена закрытия': data['Close'].round(2),
        'Объём': data['Volume'],
        'Изменение %': ((data['Close'] - data['Open']) / data['Open'] * 100).round(2)
    })
    
    return daily_data


def save_to_csv(data, ticker, filename=None):
    if filename is None:
        filename = f"{ticker}_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
    data.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Данные сохранены в файл: {filename}")


def get_multiple_tickers_daily(tickers_list, period='1y'):
    results = {}
    for ticker in tickers_list:
        try:
            data = get_daily_data(ticker, period)
            results[ticker] = data
            print(f"Успешно получены данные для {ticker}")
        except Exception as e:
            print(f"Ошибка при получении данных для {ticker}: {str(e)}")
    return results


aapl_data = get_daily_data('AMZN')

save_to_csv(aapl_data, 'AMZN')
