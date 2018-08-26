import pandas as pd
import itertools as it
import numpy as np

TOP_SECTORS = ['Business Services', 'Chemicals', 'Communication Equipment',
    'Communication Services', 'Consumer Packaged Goods', 'Drug Manufacturers',
    'Entertainment', 'Financial Services', 'Industrial Products', 'Insurance',
    'Manufacturing - Apparel & Furniture', 'Medical', 'Oil and Gas', 'REITs',
    'Retail - Apparel & Specialty', 'Technology', 'Utilites']

SECTOR_MAPPING = {
    'Advertising & Marketing Services': None,
    'Aerospace & Defense': None,
    'Agriculture': None,
    'Airlines': None,
    'Application Software': None,
    'Asset Management': 'Financial Services',
    'Autos': None,
    'Banks': 'Financial Services',
    'Beverages - Alcoholic': 'Beverages',
    'Beverages - Non-Alcoholic': 'Beverages',
    'Biotechnology': None,
    'Brokers & Exchanges': 'Financial Services',
    'Building Materials': None,
    'Business Services': None,
    'Chemicals': None,
    'Coal': None,
    'Communication Equipment': None,
    'Communication Services': None,
    'Computer Hardware': 'Technology',
    'Conglomerates': None,
    'Consulting': None,
    'Consulting & Outsourcing': 'Consulting',
    'Consumer Packaged Goods': None,
    'Credit Services': 'Financial Services',
    'Drug Manufacturers': None,
    'Education': None,
    'Employment Services': None,
    'Engineering & Construction': 'Construction',
    'Entertainment': None,
    'Farm & Construction Machinery': None,
    'Forest Products': None,
    'Health Care Plans': 'Health Care',
    'Health Care Providers': 'Health Care',
    'Homebuilding & Construction': 'Construction',
    'Industrial Distribution': None,
    'Industrial Products': None,
    'Insurance': None,
    'Insurance - Life': 'Insurance',
    'Insurance - Property & Casualty': 'Insurance',
    'Insurance - Specialty': 'Insurance',
    'Manufacturing - Apparel & Furniture': None,
    'Medical Devices': 'Medical',
    'Medical Diagnostics & Research': 'Medical',
    'Medical Distribution': 'Medical',
    'Medical Instruments & Equipment': 'Medical',
    'Metals & Mining': None,
    'Oil & Gas - Drilling': 'Oil and Gas',
    'Oil & Gas - E&P': 'Oil and Gas',
    'Oil & Gas - Integrated': 'Oil and Gas',
    'Oil & Gas - Midstream': 'Oil and Gas',
    'Oil & Gas - Refining & Marketing': 'Oil and Gas',
    'Oil & Gas - Services': 'Oil and Gas',
    'Online Media': None,
    'Packaging & Containers': None,
    'Personal Services': None,
    'Publishing': None,
    'REITs': None,
    'Real Estate Services': None,
    'Restaurants': None,
    'Retail - Apparel & Specialty': None,
    'Retail - Defensive': None,
    'Semiconductors': 'Technology',
    'Steel': None,
    'Tobacco Products': None,
    'Transportation & Logistics': None,
    'Travel & Leisure': None,
    'Truck Manufacturing': None,
    'Utilities - Independent Power Producers': 'Utilities',
    'Utilities - Regulated': 'Utilities',
    'Waste Management': None}


## Only keep close price
def filter_prices(data_filename='../../data/external/WIKI_PRICE.csv'):
    data = pd.read_csv(data_filename)
    data_filtered = data[['ticker', 'date', 'close']]
    data_filtered.to_csv('../../data/interim/WIKI_PRICE.csv', index=False) # required otherwise takes too long
    print("Filterd close price of wiki price")
    return data_filtered


def append_sector_to_company(data=None):
    if data is None:
        data = pd.read_csv('../../data/interim/WIKI_PRICE.csv')
    company_data = pd.read_csv('../../data/external/company_fundamentals.csv')
    company_data.columns = [e.lower() for e in company_data.columns]
    data_enhanced = pd.merge(data, company_data, how='left', on='ticker')
    return data_enhanced


def compute_returns(df, ticker):
    ts = df[['date', 'close']]
    ts = ts.set_index('date')
    ts = ts.pct_change()
    ts = ts.reset_index()
    df = df.reset_index()
    ts = pd.concat([ts, df[['name', 'sector']]], axis=1)
    ts['ticker'] = ticker
    return ts


def create_returns(data=None):
    data_returns = []
    for c, g in data.groupby('ticker'):
        df = compute_returns(g, c)
        data_returns.append(df)
    df_returns = pd.concat(data_returns)
    return df_returns


def remap_sectors(data, sector_mapping=SECTOR_MAPPING):
    sector_mapping = {k: v for k, v in sector_mapping.items() if v}
    data = data.replace({'sector': sector_mapping})
    return data


def keep_top_sectors(data, top_sectors=TOP_SECTORS):
    data = data.set_index('sector')
    data = data.loc[top_sectors].reset_index()
    return data


def index_construction(data_returns):
    df = data_returns.set_index('sector')
    df_mean = df.groupby(['sector', 'date']).agg('mean')

    # We whould do a weighted average with market cap but we don't have the
    # data so we are going to restrict the min and max returns to avoid absurd
    # result
    min_max_range = lambda x: max(min(x, 0.05),- 0.05)
    df_mean = df_mean.applymap(min_max_range)
    df_mean = df_mean.unstack('sector')

    indices_returns = df_mean
    idx = (indices_returns.index == '1990-01-03').tolist() # index start time
    start_idx = list(it.compress(range(len(idx)), idx))[0]

    indices_returns.columns = indices_returns.columns.droplevel()
    indices_values = np.log(1+indices_returns).cumsum()
    indices_values = indices_values - indices_values.iloc[start_idx]
    return indices_returns, indices_values


if __name__ == '__main__':
    print("Filtering the WIKI Price data source")
    data_filtered = filter_prices()
    print("Appending sectors")
    data_sectors = append_sector_to_company(data_filtered)
    print("Computing returns per ticker")
    data_returns = create_returns(data_sectors)
    print("Filtering the sectors")
    data_returns_top_sector = keep_top_sectors(remap_sectors(data_returns))
    print("Creating the indices")
    data_indices_returns, data_indices_level = index_construction(data_returns_top_sector)

    print("Saving data")
    print("Saving stock returns")
    data_returns_top_sector.to_csv('../../data/processed/wiki_stocks_returns.csv')
    print("Saving indices returns")
    data_indices_returns.to_csv('../../data/processed/wiki_indices_returns.csv')
    print("Saving levels returns")
    data_indices_level.to_csv('../../data/processed/wiki_indices_level.csv')
