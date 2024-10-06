# model/script.py
import pandas as pd
import os

# Get the absolute path to the data directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'census_cleaned.csv')

# Read the CSV file
df = pd.read_csv(data_path)

# Display the DataFrame
print(df.head())
