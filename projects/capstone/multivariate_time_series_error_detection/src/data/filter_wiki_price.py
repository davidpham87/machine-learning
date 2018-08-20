import pandas as pd


# only keep close
data = pd.read_csv('../../data/external/WIKI_PRICE.csv')
data = data[['ticker', 'close']]
data.to_csv('../../data/interim/WIKI_PRICE.csv')



