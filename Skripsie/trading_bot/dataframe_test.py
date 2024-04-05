import polars as pl
import json

def parse_strings_to_dataframe(string_list):
    # Initialize empty lists to store parsed data
    timestamps = []
    opens = []
    closes = []
    highs = []
    lows = []
    volumes = []

    # Iterate through each string in the list
    for string_data in string_list:
        # Convert the string to a dictionary
        data_dict = json.loads(string_data)
        
        # Extract data from the dictionary and append to respective lists
        timestamps.append(data_dict['timestamp'])
        opens.append(float(data_dict['open']))
        closes.append(float(data_dict['close']))
        highs.append(float(data_dict['high']))
        lows.append(float(data_dict['low']))
        volumes.append(float(data_dict['volume']))
    
    # Create a Polars DataFrame
    df = pl.DataFrame({
        'timestamp': pl.Int64Series(timestamps),
        'open': pl.Float64Series(opens),
        'close': pl.Float64Series(closes),
        'high': pl.Float64Series(highs),
        'low': pl.Float64Series(lows),
        'volume': pl.Float64Series(volumes)
    })
    
    return df


def format_string_with_double_quotes(input_string):
    # Convert string to dictionary
    data_dict = eval(input_string)

    # Convert dictionary to JSON string with double quotes
    formatted_string = json.dumps(data_dict)

    return formatted_string

# Example usage:
input_string = "{'timestamp': 1366794000000, 'open': '1600.00', 'close': '1600.00', 'high': '1600.00', 'low': '1600.00', 'volume': '0.00'}"
formatted_string = format_string_with_double_quotes(input_string)
print(formatted_string)

# Example usage:
string_list = [
    "{'timestamp': 1366792200000, 'open': '1600.00', 'close': '1600.00', 'high': '1600.00', 'low': '1600.00', 'volume': '0.00'}",
    "{'timestamp': 1366792300000, 'open': '1610.00', 'close': '1620.00', 'high': '1630.00', 'low': '1605.00', 'volume': '10.00'}"
]
print(string_list)
formated_list = format_string_with_double_quotes(string_list)
print('\n'+ formated_list )
#dataframe = parse_strings_to_dataframe(string_list)
#print(dataframe)

