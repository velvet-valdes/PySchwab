import PySchwab as stonks

# Token Usage - find a way to figure out expiry?
token = stonks.get_bearer_token()

# List of stock symbols you want to query - TODO web scraping
symbols = stonks.csv_to_list('input', 500)

# Create main data frame from ticker symbols
main_frame = stonks.fetch_and_aggregate_data(symbols,token)

# Process data types for data frame
stonks.prepare_data_frame(main_frame)

# Summarize the data
summary = main_frame.describe(include='all')

