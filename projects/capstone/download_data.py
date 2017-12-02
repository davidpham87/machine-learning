import quandl
import pandas as pd

quandl.ApiConfig.api_key = "GMRJ3WPfaDvbxReufnku"

data = quandl.get_table('WIKI/PRICES', paginate=True)
