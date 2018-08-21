import pandas as pd


# only keep close
def filter_prices():
    data = pd.read_csv('../../data/external/WIKI_PRICE.csv')
    data = data[['ticker', 'close']]
    data.to_csv('../../data/interim/WIKI_PRICE.csv')
    print("Filterd close price of wiki price")
    return data

def append_sector_to_company(data):

    return

company_data = pd.read_csv('../../data/external/company_fundamentals.csv')
company_data.columns = [e.lower() for e in company_data.columns]


data_enhanced = data.join(company_data, 'ticker', 'left', '_share_price', '_company_data')


data_enhanced.dropna()
