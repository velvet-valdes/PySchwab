import math
import os
import pandas as pd
import requests
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from base64 import b64encode


def json_to_dataframe(json_data):
    """
    Converts a JSON object containing multiple ticker symbols into a pandas DataFrame.

    The function processes each ticker symbol's data, flattens any nested structures,
    and then compiles all the data into a DataFrame. Each row in the DataFrame corresponds
    to a ticker symbol, and columns represent the various data points associated with each symbol.

    Args:
        json_data (dict): JSON object containing data for multiple ticker symbols.

    Returns:
        pd.DataFrame: A DataFrame containing all the data from the JSON object, with one row per symbol.

    Example:
        >>> json_data = {
        ...     "AAPL": {"price": 150, "volume": 10000},
        ...     "GOOG": {"price": 2700, "volume": 2000}
        ... }
        >>> df = json_to_dataframe(json_data)
        >>> print(df.head())
           price  volume symbol
        0  150     10000  AAPL
        1  2700     2000  GOOG
    """
    # Initialize an empty list to collect rows
    rows = []

    # Process each ticker symbol in the JSON data
    for symbol, data in json_data.items():
        # Flatten the nested structure and prepend prefix to nested keys
        def flatten(data, prefix=""):
            items = {}
            for key, value in data.items():
                new_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    items.update(flatten(value, f"{new_key}_"))
                else:
                    items[new_key] = value
            return items

        # Flatten the current ticker's data and add the symbol
        row = flatten(data)
        row["symbol"] = symbol  # Make sure the symbol is included as part of the row

        # Append the row dictionary to the list
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    # Set a consistent order for DataFrame columns
    column_order = sorted(df.columns)
    df = df[column_order]

    return df


def csv_to_list(directory, list_length=10):
    """
    Reads multiple CSV files from the specified directory and returns a list of lists,
    with each inner list containing stock ticker symbols up to the specified length.

    Args:
        directory (str): The directory containing CSV files.
        list_length (int): The desired length of each inner list of ticker symbols. Defaults to 10.

    Returns:
        list[list[str]]: A list where each inner list contains ticker symbols up to the specified length.

    Example:
        >>> csv_to_list('/path/to/csv/files', list_length=5)
        [['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN'], ['FB', 'NFLX', 'NVDA']]
    """
    all_tickers = []

    # List all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            # Read the CSV file
            data = pd.read_csv(file_path)
            # Assuming the ticker symbols are in the first column
            tickers = data.iloc[:, 0].tolist()

            # Split the tickers into chunks of specified length
            for i in range(0, len(tickers), list_length):
                all_tickers.append(tickers[i : i + list_length])

    return all_tickers


def get_bearer_token():
    """
    Retrieves a bearer token from the Schwab API using client credentials.

    The function uses the client ID and client secret stored in environment variables
    to request an OAuth 2.0 bearer token from the Schwab API.

    Returns:
        str: The OAuth 2.0 bearer token as a string.

    Raises:
        requests.exceptions.HTTPError: If the request for the bearer token fails.

    Example:
        >>> token = get_bearer_token()
        >>> print(token)
        'eyJhbGciOiJSUzI1NiIs...'
    """
    # Retrieve client ID and client secret from environment variables
    client_id = os.getenv("SCHWAB_CLIENT_ID")
    client_secret = os.getenv("SCHWAB_CLIENT_SECRET")

    # Encode the client credentials in Base64 for the Basic Auth header
    client_credentials = f"{client_id}:{client_secret}"
    encoded_credentials = b64encode(client_credentials.encode()).decode("utf-8")

    # Token URL provided by Schwab API
    token_url = "https://api.schwabapi.com/v1/oauth/token"

    # Headers for the POST request
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {encoded_credentials}",
    }

    # Payload for the POST request
    payload = {"grant_type": "client_credentials"}

    # Make the POST request to get the token
    response = requests.post(token_url, headers=headers, data=payload)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # Extract the token from the response
    token_data = response.json()
    access_token = token_data.get("access_token")

    return access_token


def fetch_and_aggregate_data(symbols, token):
    """
    Fetches and aggregates market data for a list of stock symbols.

    This function makes API calls to the Schwab API to retrieve market data
    for each stock symbol provided, aggregates the results, and returns them
    as a Pandas DataFrame.

    Args:
        symbols (list[list[str]]): A list of lists, where each inner list contains stock symbols.
        token (str): The bearer token for API authentication.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated market data.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the API request.

    Example:
        >>> symbols = [['AAPL', 'GOOG'], ['MSFT', 'TSLA']]
        >>> token = get_bearer_token()
        >>> df = fetch_and_aggregate_data(symbols, token)
        >>> print(df.head())
    """
    base_url = "https://api.schwabapi.com/marketdata/v1/quotes"
    combined_responses = {}

    for symbol in symbols:
        # Filter out 'nan' values and prepare the symbols for the API call
        filtered_symbol = ",".join(
            str(s) for s in symbol if not (isinstance(s, float) and math.isnan(s))
        )

        params = {"indicative": "false", "symbols": filtered_symbol}
        headers = {"Authorization": f"Bearer {token}"}

        try:
            # Make the API call
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()  # Check for HTTP request errors

            # Parse the JSON from the response
            json_response = response.json()

            # Update the combined_responses dictionary with the new data
            combined_responses.update(json_response)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    # Convert the aggregated JSON data into a DataFrame
    return json_to_dataframe(combined_responses)


def create_pairwise_plot(df):
    """
    Creates a pairwise plot of numeric columns in the provided DataFrame.

    The function filters the DataFrame to include only numeric columns and
    then generates a pairwise plot using Seaborn to visualize the relationships
    between these columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be plotted.

    Returns:
        None

    Example:
        >>> data = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [5, 6, 7, 8],
        ...     'C': ['a', 'b', 'c', 'd']
        ... })
        >>> create_pairwise_plot(data)
    """
    # Filter columns to include only numeric types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns to plot.")
        return

    # Create pairwise plot
    sns.pairplot(df[numeric_cols])
    plt.show()


def prepare_data_frame(df):
    """
    Prepares the DataFrame for analysis by performing several key transformations:

    1. Sets the DataFrame index to 'symbol' if it exists.
    2. Converts specific columns to appropriate data types (e.g., datetime, boolean, category).
    3. Handles special formatting for certain columns like 'ssid' and large integers.
    4. Creates dummy variables for specified categorical columns, while handling cases
       where a column has only one unique value.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        None: The function modifies the DataFrame in place.

    Example:
        >>> data = pd.DataFrame({
        ...     'symbol': ['AAPL', 'GOOG'],
        ...     'fundamental_declarationDate': ['2024-08-01', '2024-08-05'],
        ...     'reference_isHardToBorrow': [True, False],
        ...     'reference_otcMarketTier': ['OTCQB', 'OTCQX'],
        ...     'reference_htbRate': [1.5, 2.1]
        ... })
        >>> prepare_data_frame(data)
    """
    # Set index for the data frame
    if "symbol" in df.columns:
        df.set_index("symbol", inplace=True)

    # Convert date columns to datetime
    date_columns = [
        "fundamental_declarationDate",
        "fundamental_divExDate",
        "fundamental_divPayDate",
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Convert boolean columns
    boolean_columns = ["reference_isHardToBorrow", "reference_isShortable"]
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # Convert 'ssid' to string to preserve any special formatting or leading zeros
    if "ssid" in df.columns:
        df["ssid"] = df["ssid"].astype(str)

    # Convert market tier data to category type for efficiency
    if "reference_otcMarketTier" in df.columns:
        df["reference_otcMarketTier"] = df["reference_otcMarketTier"].astype("category")

    # Convert rates and other similar financial metrics to float
    if "reference_htbRate" in df.columns:
        df["reference_htbRate"] = df["reference_htbRate"].astype(float)

    # Handle large integers with potential NaN values using nullable integer type
    if "reference_htbQuantity" in df.columns:
        df["reference_htbQuantity"] = df["reference_htbQuantity"].astype("Int64")

    # Identify specific categorical columns that need dummy variables
    categorical_columns_to_dummy = [
        "assetSubType",
        "assetMainType",
    ]  # Add other column names as needed

    # Check for columns with only one unique value and handle dummy variable creation
    for col in categorical_columns_to_dummy:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            # Convert all unique values to strings for consistent sorting
            unique_values_str = sorted(map(str, unique_values))
            print(f"""Indicator Variables:
Processing column '{col}' with unique values:
{unique_values_str}
                """)

            if len(unique_values_str) > 1:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                reference_category = unique_values_str[
                    0
                ]  # The first value in sorted unique values is the reference
                print(f"""
Dropping reference category '{reference_category}' for column '{col}'
                """)
                df.drop(columns=[col], inplace=True)
                for dummy_col in dummies.columns:
                    df[dummy_col] = dummies[dummy_col].astype(int)
            else:
                # Handle columns with a single unique value
                unique_value = unique_values_str[0]
                print(f"""
Column '{col}' has a single unique value: '{unique_value}', creating a constant dummy variable
                """)
                df[f"{col}_is_{unique_value}"] = 1
                df.drop(columns=[col], inplace=True)


def log_transform(df, columns):
    """
    Applies a natural logarithm transformation to specified numeric columns in the DataFrame.

    The function creates new columns in the DataFrame with the '_log' suffix for each
    specified column. Any zero values in the original columns are replaced with NaN
    before applying the logarithm, and rows with NaN values are dropped from the
    transformed columns.

    Args:
        df (pd.DataFrame): The original DataFrame containing the data to be transformed.
        columns (list[str]): A list of column names to apply the logarithm transformation to.

    Returns:
        pd.DataFrame: A new DataFrame with the original data and the newly transformed columns.

    Example:
        >>> data = pd.DataFrame({
        ...     'A': [1, 10, 100],
        ...     'B': [0, 5, 20]
        ... })
        >>> transformed_data = log_transform(data, ['A', 'B'])
        >>> print(transformed_data.head())
           A   B  A_log  B_log
        0  1   0    0.0    NaN
        1  10  5    2.3    1.6
        2  100 20   4.6    3.0
    """
    transformed_df = df.copy()
    for col in columns:
        if np.issubdtype(df[col].dtype, np.number):
            transformed_df[col + "_log"] = np.log(df[col].replace(0, np.nan)).dropna()
    return transformed_df


def log_transform_all(df):
    """
    Applies a natural logarithm transformation to all numeric columns in the DataFrame
    that contain only positive values.

    The function creates new columns in the DataFrame with the '_log' suffix for each
    numeric column that is eligible for logarithm transformation. Columns with non-positive
    values are skipped.

    Args:
        df (pd.DataFrame): The original DataFrame containing the data to be transformed.

    Returns:
        pd.DataFrame: A new DataFrame with the original data and the newly transformed columns.

    Example:
        >>> data = pd.DataFrame({
        ...     'A': [1, 10, 100],
        ...     'B': [0, 5, 20],
        ...     'C': [2, 3, 4]
        ... })
        >>> transformed_data = log_transform_all(data)
        >>> print(transformed_data.head())
           A   B  C  A_log  C_log
        0  1   0  2   0.0    0.7
        1  10  5  3   2.3    1.1
        2  100 20  4   4.6    1.4
    """
    transformed_df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"Processing column: {col}")  # Debugging: print column name
        if (df[col] > 0).all():  # Check if all values are positive
            print(f"Transforming column: {col}")  # Debugging: print transforming column
            transformed_df[col + "_log"] = np.log(df[col])
        else:
            print(
                f"Skipping column: {col} (contains non-positive values)"
            )  # Debugging: print skipped column
    return transformed_df


def square_root_transform(df, columns):
    pass  # use this function to apply changes in place


def inverse_square_root_transform(df, columns):
    pass  # use this function to apply changes in place


def zscore_transform(df):
    # Perform Z-score standardization on numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64", "Int64"]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    return df
