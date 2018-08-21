import pandas as pd


# only keep close
def filter_prices():
    data = pd.read_csv('../../data/external/WIKI_PRICE.csv')
    data_filtered = data[['ticker', 'date', 'close']]
    data_filtered.to_csv('../../data/interim/WIKI_PRICE.csv', index=False)
    print("Filterd close price of wiki price")
    return data_filtered



def append_sector_to_company(data=None):
    if data is None:
        data = pd.read_csv('../../data/interim/WIKI_PRICE.csv')
    company_data = pd.read_csv('../../data/external/company_fundamentals.csv')
    company_data.columns = [e.lower() for e in company_data.columns]
    data_enhanced = pd.merge(data, company_data, how='left', on='ticker')
    data_enhanced.to_csv('../../data/interim/WIKI_PRICE_SECTOR.csv', index=False)
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
    if data is None:
        data = pd.read_csv('../../data/interim/WIKI_PRICE_SECTOR.csv')
    data_returns = []
    for c, g in data.groupby('ticker'):
        print(c)
        df = compute_returns(g, c)
        data_returns.append(df)
    df_returns = pd.concat(data_returns)
    df_returns.to_csv('../../data/processed/wiki_returns_sector.csv', index=false)
    return df_returns

sector_mapping = {
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
    'Consulting': 'Consulting',
    'Consulting & Outsourcing': 'Consulting',
    'Consumer Packaged Goods': None,
    'Credit Services': 'Financial Services',
    'Drug Manufacturers': None,
    'Education': None,
    'Employment Services': None,
    'Engineering & Construction': None,
    'Entertainment': None,
    'Farm & Construction Machinery': None,
    'Forest Products': None,
    'Health Care Plans': None,
    'Health Care Providers': None,
    'Homebuilding & Construction': None,
    'Industrial Distribution': None,
    'Industrial Products': None,
    'Insurance': 'Insurance',
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
    'Utilities - Independent Power Producers': None,
    'Utilities - Regulated': None,
    'Waste Management': None}
