import pandas as pd

def filter_range(data, column, start, end):
    return data[(data[column] < end) * (data[column] > start)]

filter_date = lambda d, s, e: filter_range(d,  'date', s, e)

x = pd.read_csv('data/adj_close.csv', index_col=False)
x.index = pd.to_datetime(x.date)


