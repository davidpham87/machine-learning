import functools
import numpy as np
import pandas as pd

from scipy.special import expit

import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
import tensorflow as tf

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


def log_softmax(x):
    return x - np.log(np.sum(np.exp(x)))


def sigmoid(x):
    return expit(x)


def load_data(stock_filename=None, indices_filename=None):

    if stock_filename is None:
        stock_filename = '../../data/processed/wiki_stocks_returns.csv'

    if indices_filename is None:
        indices_filename = '../../data/processed/wiki_indices_returns.csv'

    stocks = pd.read_csv(stock_filename, index_col=False) # long format
    indices = pd.read_csv(indices_filename, index_col=False) # wide format

    # Implementation of hierarchical clustering
    drop_column = lambda df,i=0: df.drop(df.columns[i], axis=1)

    stocks = drop_column(stocks)
    stocks = stocks.drop('name', axis=1)
    stocks = stocks.dropna()

    companies = stocks.groupby('ticker').first().reset_index()
    sectors_counts = companies.sector.value_counts()
    sectors_proportions = sectors_counts/sectors_counts.sum()
    sectors_unique = sectors_counts.index.tolist()

    stocks = stocks.set_index('ticker')

    indices_ts = df_to_ts(indices[['date'] + sectors_unique])
    stocks_ts = df_to_ts(stocks.reset_index())

    stocks_all = pd.merge(stocks_ts, indices_ts, 'left')
    stocks_all = stocks_all.dropna() # loss of 200 000 observations
    stocks_all = stocks_all.drop('sector', axis=1)
    stocks_all = stocks_all.groupby('ticker').apply(df_to_ts)
    stocks_all = stocks_all.drop(['ticker', 'date'], axis=1)
    stocks_all = stocks_all.rename(columns={'close': 'pct_return'})

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(sectors_counts.index.tolist())
    ticker_to_sector = dict(zip(companies.ticker, label_encoder.transform(companies.sector)))

    return stocks_all, companies, label_encoder, ticker_to_sector

def sectors_statistics(companies):
    sectors_counts = companies.sector.value_counts()
    sectors_proportions = sectors_counts/sectors_counts.sum()
    sectors_unique = sectors_counts.index.tolist()
    return sectors_counts, sectors_proportions, sectors_unique

def random_subset(df, window_size=21):
    idx = np.random.randint(0, df.shape[0]-window_size)
    ts = df[idx:idx+window_size]
    return ts

class StocksSequence(keras.utils.Sequence):

    def __init__(self, stocks_data,  companies_data, window_size, label_encoder,
                 batch_size, mode_key='train'):
        self.stocks_data = stocks_data
        self.batch_size = batch_size
        self.label_encoder = label_encoder
        self.companies_data = companies_data
        self.window_size = window_size
        self.mode_key = mode_key
        self.classes = []

        _, sectors_proportion, _ = sectors_statistics(companies_data)
        self.sectors_proportion = sectors_proportion

    def __len__(self):
        return int(np.ceil(self.stocks_data.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):

        idx = np.random.choice(self.companies_data.shape[0], self.batch_size)
        df = self.companies_data.iloc[idx]
        model_input_data = [random_subset(self.stocks_data.loc[t], self.window_size)
                            for t in df.ticker]
        model_input = np.array([df.values for df in model_input_data])

        if self.mode_key != 'infer':
            y_true = self.label_encoder.transform(df.sector)

        if self.mode_key == 'infer':
            if 'sector' in df.columns:
                self.classes.extend(self.label_encoder.transform(df.sector))

        if self.mode_key == 'infer':
            return model_input

        return model_input, y_true


model = keras.models.load_model('checkpoint/model_weights_seventeenth.json')


stocks_all, companies, label_encoder, ticker_to_sector = load_data(
    '../data/processed/wiki_stocks_returns.csv',
    '../data/processed/wiki_indices_returns.csv')

sectors_counts, sectors_proportions, sectors_unique = sectors_statistics(companies)
companies_data = {}
data_split = split_companies_train_dev_test(companies)
for i, k in enumerate(['train', 'dev', 'test']):
    companies_data[k] = data_split[i]
stocks_data = {k: filter_stocks(stocks_all, v.ticker) for k, v in companies_data.items()}


sequence_generator_test = StocksSequence(
    stocks_data['test'], companies_data['test'], 63, label_encoder, 256, 'eval')
y = model.evaluate_generator(sequence_generator_test, steps=100)

sequence_generator_infer = StocksSequence(
    stocks_data['test'], companies_data['test'], 63, label_encoder, 256, 'infer')

prediction = model.predict_generator(sequence_generator_infer, 1000)
y_pred = np.argmax(prediction, 1)
y_true = np.array(sequence_generator_infer.classes[:-256])
conf_mat = confusion_matrix(y_true, y_pred)

conf_df = pd.DataFrame(conf_mat, columns=label_encoder.classes_.tolist(), index=label_encoder.classes_.tolist())
conf_df.to_csv('confusion_matrix.csv')
conf_df_ratio = conf_df.apply(lambda x: 100*round(x/x.sum(), 4), 0)
conf_df_ratio.to_csv('confusion_matrix_ratio.csv')

print(classification_report(y_true, y_pred, target_names=label_encoder.classes_.tolist()))
