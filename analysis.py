import PySchwab as stonks
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
    "fundamental_eps",
    "fundamental_divYield",
    "fundamental_peRatio",
    "fundamental_divPayAmount",
    "quote_52WeekLow",
    "quote_52WeekHigh",
    "regular_regularMarketLastPrice",
]
sub_frame = main_frame[sub_frame_columns]

# Transform the data frame
# sub_frame = stonks.zscore_transform(sub_frame)

# # Summarize the data
summary = sub_frame.describe(include="all")

# Subset
ceiling = 500
floor = 0

# Temp frame for testing
tmp_frame = sub_frame[
    (sub_frame["regular_regularMarketLastPrice"] <= ceiling)
    & (sub_frame["regular_regularMarketLastPrice"] >= floor)
]

# Generate the correlation matrix
corr = sub_frame.corr()

# Plot the heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Box Plots
sns.boxplot(data=tmp_frame["regular_regularMarketLastPrice"])
plt.title("Box Plot of Each Column")
plt.show()

# # Histogram
# sns.histplot(tmp_frame['regular_regularMarketLastPrice'], kde=True)
# plt.show()

# Number of observations
N = len(tmp_frame["regular_regularMarketLastPrice"])

# Sturges' formula for number of bins
num_bins_sturges = int(np.ceil(np.log2(N) + 1))

# Freedman-Diaconis rule for bin width
IQR = np.percentile(tmp_frame["regular_regularMarketLastPrice"], 75) - np.percentile(
    tmp_frame["regular_regularMarketLastPrice"], 25
)
bin_width_fd = 2 * IQR / np.cbrt(N)
num_bins_fd = int(
    (
        tmp_frame["regular_regularMarketLastPrice"].max()
        - tmp_frame["regular_regularMarketLastPrice"].min()
    )
    / bin_width_fd
)

# Plot using the Sturges' formula
sns.histplot(
    tmp_frame["regular_regularMarketLastPrice"], bins=num_bins_sturges, kde=True
)
plt.title("Histogram with Sturges' Formula")
plt.show()

# Plot using the Freedman-Diaconis rule
sns.histplot(tmp_frame["regular_regularMarketLastPrice"], bins=num_bins_fd, kde=True)
plt.title("Histogram with Freedman-Diaconis Rule")
plt.show()
