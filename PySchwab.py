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

    Args:
        json_data (dict): JSON object with multiple ticker symbol data.

    Returns:
        pd.DataFrame: DataFrame containing all the data from the JSON object.
    """
    # Initialize an empty list to collect rows
    rows = []

    # Process each ticker symbol in the JSON data
    for symbol, data in json_data.items():
        # Flatten the nested structure and prepend prefix to nested keys
        def flatten(data, prefix=''):
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
        row['symbol'] = symbol  # Make sure the symbol is included as part of the row

        # Append the row dictionary to the list
        rows.append(row)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)
    
    # Set a consistent order for DataFrame columns, if necessary
    column_order = sorted(df.columns)  # Optional: Define a specific order based on your needs
    df = df[column_order]

    return df

def csv_to_list(directory, list_length=10):
    """
    Reads multiple CSV files from the specified directory and returns a list of lists,
    with each inner list containing stock ticker symbols up to the specified length.

    Args:
    directory (str): The directory containing CSV files.
    list_length (int): The desired length of each inner list of ticker symbols.

    Returns:
    list of lists: A list where each inner list contains ticker symbols up to the specified length.
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
                all_tickers.append(tickers[i:i + list_length])

    return all_tickers

def get_bearer_token():
    # Retrieve client ID and client secret from environment variables
    client_id = os.getenv('SCHWAB_CLIENT_ID')
    client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
    
    # Encode the client credentials in Base64 for the Basic Auth header
    client_credentials = f"{client_id}:{client_secret}"
    encoded_credentials = b64encode(client_credentials.encode()).decode('utf-8')
    
    # Token URL provided by Schwab API
    token_url = 'https://api.schwabapi.com/v1/oauth/token'
    
    # Headers for the POST request
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_credentials}'
    }
    
    # Payload for the POST request
    payload = {
        'grant_type': 'client_credentials'
    }
    
    # Make the POST request to get the token
    response = requests.post(token_url, headers=headers, data=payload)
    
    # Raise an exception if the request was unsuccessful
    response.raise_for_status()
    
    # Extract the token from the response
    token_data = response.json()
    access_token = token_data.get('access_token')
    
    return access_token

def fetch_and_aggregate_data(symbols, token):
    """
    Fetches and aggregates market data for a list of stock symbols.

    Parameters:
        symbols (list): A list of lists containing stock symbols.
        token (str): The bearer token for API authentication.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated data.
    """
    base_url = "https://api.schwabapi.com/marketdata/v1/quotes"
    combined_responses = {}

    for symbol in symbols:
        # Filter out 'nan' values and prepare the symbols for the API call
        filtered_symbol = ','.join(str(s) for s in symbol if not (isinstance(s, float) and math.isnan(s)))

        params = {
            'indicative': 'false',
            'symbols': filtered_symbol
        }
        headers = {
            'Authorization': f'Bearer {token}'
        }

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
    Sets the DataFrame index, converts specific columns to appropriate data types,
    ensures that the data frame columns have the correct types for further analysis,
    performs Z-score standardization, and creates dummy variables for specified categorical columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        None: Modifies the DataFrame in place.
    """
    # Set index for the data frame
    if 'symbol' in df.columns:
        df.set_index('symbol', inplace=True)

    # Convert date columns to datetime
    date_columns = ['fundamental_declarationDate', 'fundamental_divExDate', 'fundamental_divPayDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Convert boolean columns
    boolean_columns = ['reference_isHardToBorrow', 'reference_isShortable']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype('boolean')

    # Convert 'ssid' to string to preserve any special formatting or leading zeros
    if 'ssid' in df.columns:
        df['ssid'] = df['ssid'].astype(str)

    # Convert market tier data to category type for efficiency
    if 'reference_otcMarketTier' in df.columns:
        df['reference_otcMarketTier'] = df['reference_otcMarketTier'].astype('category')

    # Convert rates and other similar financial metrics to float
    if 'reference_htbRate' in df.columns:
        df['reference_htbRate'] = df['reference_htbRate'].astype(float)

    # Handle large integers with potential NaN values using nullable integer type
    if 'reference_htbQuantity' in df.columns:
        df['reference_htbQuantity'] = df['reference_htbQuantity'].astype('Int64')

    # Perform Z-score standardization on numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

    # Identify specific categorical columns that need dummy variables
    categorical_columns_to_dummy = ['assetSubType', 'assetMainType']  # Add other column names as needed

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
                reference_category = unique_values_str[0]  # The first value in sorted unique values is the reference
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
                df[f'{col}_is_{unique_value}'] = 1
                df.drop(columns=[col], inplace=True)

def log_transform(df, columns):
    transformed_df = df.copy()
    for col in columns:
        if np.issubdtype(df[col].dtype, np.number):
            transformed_df[col + '_log'] = np.log(df[col].replace(0, np.nan)).dropna()
    return transformed_df

def log_transform_all(df):
    transformed_df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"Processing column: {col}")  # Debugging: print column name
        if (df[col] > 0).all():  # Check if all values are positive
            print(f"Transforming column: {col}")  # Debugging: print transforming column
            transformed_df[col + '_log'] = np.log(df[col])
        else:
            print(f"Skipping column: {col} (contains non-positive values)")  # Debugging: print skipped column
    return transformed_df