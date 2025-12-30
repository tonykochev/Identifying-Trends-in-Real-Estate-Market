import pandas as pd

# Load the CSV file
df = pd.read_csv('Leases.csv')

# Ensure column names are lowercase (optional)
df.columns = df.columns.str.lower()

# Filter with partial match using regex and case-insensitive search
filtered_df = df[
    (df['leasedsf'] >= 10000) &
    (df['internal_industry'].str.contains('tech|legal|financial', case=False, na=False)) &
    (df['internal_market_cluster'].notna()) &
    (df['internal_market_cluster'].str.strip() != '')
]

# Preview filtered data
print(filtered_df.head())

# Optionally save it
filtered_df.to_csv('filtered_leases.csv', index=False)
