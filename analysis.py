import PySchwab as stonks
import seaborn as sns
import matplotlib.pyplot as plt


# Token Usage - find a way to figure out expiry?
token = stonks.get_bearer_token()

# Testing symbol list
# symbols = [
#     ["AAPL", "GOOG", "AVGO", "NVDA", "INTC", "AMD", "MSFT", "QCOM", "PTC", "AMZN"]
# ]

# List of stock symbols you want to query - TODO web scraping
symbols = stonks.csv_to_list("input", 500)


# Create main data frame from ticker symbols
main_frame = stonks.fetch_and_aggregate_data(symbols, token)

# Process data types for data frame
stonks.prepare_data_frame(main_frame)

# Subset the main frame
sub_frame_columns = [
    "fundamental_avg10DaysVolume",
    "regular_regularMarketLastPrice",
    "fundamental_eps",
    "fundamental_divYield",
    "fundamental_peRatio",
    "fundamental_divPayAmount",
    "quote_52WeekHigh",
    "quote_52WeekLow"
]
sub_frame = main_frame[sub_frame_columns]

# Transform the data frame
sub_frame = stonks.zscore_transform(sub_frame)

# Summarize the data
summary = sub_frame.describe(include="all")

# Generate the correlation matrix
corr = sub_frame.corr()

# Plot the heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()
