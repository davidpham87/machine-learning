# import quandl
# quandl.ApiConfig.api_key = "GMRJ3WPfaDvbxReufnku"
# data = quandl.get_table('WIKI/PRICES', paginate=True)

import pandas as pd

data = pd.read_csv('data/prices.csv')

# Take adjusted end of day prices
data_adj_close = data[['ticker', 'date', 'adj_close']]
data_adj_close.to_csv('data/adj_close.csv')


data_close = data[['ticker', 'date', 'close']]
data_close.to_csv('data/close.csv')

x = data



