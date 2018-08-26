import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import silouhette_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def split_companies_train_dev_test(companies):
    "Return train, dev, test set for companies"
    train, test = train_test_split(companies, test_size=0.1, stratify = companies.sector)
    train, dev = train_test_split(train, test_size=0.1, stratify = train.sector)
    return train, dev, test


def filter_stocks(stocks, tickers):
    return stocks.loc[tickers]


def df_to_ts(df):
    res = df.copy()
    res.index = pd.DatetimeIndex(pd.to_datetime(res.date))
    res.drop('date', axis=1)
    return res

# Make train dev test set.
np.random.seed(42)

### Feature engineering

# long format
stocks = pd.read_csv('../../data/processed/wiki_stocks_returns.csv', index_col=False)

# wide format
indices = pd.read_csv('../../data/processed/wiki_indices_returns.csv', index_col=False)

# Implementation of hierarchical clustering
drop_column = lambda df,i=0: df.drop(df.columns[i], axis=1)

stocks = drop_column(stocks)
stocks = stocks.drop('name')
stocks = stocks.dropna()

stocks = stocks.set_index('ticker')
stocks_data = {k: filter_stocks(stocks, v.ticker) for k, v in companies_data.items()}

indices_ts = df_to_ts(indices[['date'] + sectors_unique])
stocks_ts = df_to_ts(stocks.reset_index())

stocks_all = pd.merge(stocks_ts, indices_ts, 'left')
stocks_all = stocks_all.dropna() # loss of 200 000 observations
stocks_all = stocks_all.drop('sector', axis=1)
stocks_all = stocks_all.groupby('ticker').apply(df_to_ts)
stocks_all = stocks_all.drop(['ticker', 'date'], axis=1)

# Baseline model
# mean values

companies = stocks.groupby('ticker').first().reset_index()

sectors_counts = companies.sector.value_counts()
sectors_proportions = sectors_counts/sectors_counts.sum()
sectors_unique = sectors_counts.index.tolist()

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(sectors_counts.index.tolist())
ticker_to_sector = dict(zip(companies.ticker, label_encoder.transform(companies.sector)))

max_proportion_baseline = sectors_proportions.max()
biggest_sector = sectors_proportions.argmax()

print("Most representated class:", biggest_sector, ', with proportion of ', round(100*max_proportion_baseline, 2), '%.')
# Accuracy of our models should be better than max_proportion_baseline.

# Second baseline, take the most similar class
stocks_all.set_index('ticker')


companies_data = {}
data_split = split_companies_train_dev_test(companies)
for i, k in enumerate(['train', 'dev', 'test']):
    companies_data[k] = data_split[i]


## Create 2 months period so around 21*2=42 dataframes so the 42x17

