import PySchwab as stonks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn_pandas import DataFrameMapper
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# Token Usage - find a way to figure out expiry?
token = stonks.get_bearer_token()

# List of stock symbols you want to query - TODO web scraping
symbols = stonks.csv_to_list('input', 500)

# Create main data frame from ticker symbols
main_frame = stonks.fetch_and_aggregate_data(symbols,token)

# # Process data types for data frame
# stonks.prepare_data_frame(main_frame)

# # Summarize the data
# summary = main_frame.describe(include='all')

# # TODO create pandas filters for dataframe 
# band_phylter_00 = (main_frame['quote_closePrice'] <= 1 ) & (main_frame['quote_closePrice'] >=0)
# band_phylter_01 = (main_frame['quote_closePrice'] <= 2 ) & (main_frame['quote_closePrice'] >=1)
# band_phylter_02 = (main_frame['quote_closePrice'] <= 3 ) & (main_frame['quote_closePrice'] >=2)

