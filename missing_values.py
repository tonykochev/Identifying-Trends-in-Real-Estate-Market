import pandas as pd

# Load the CSV
df = pd.read_csv('filtered_leases.csv')

# Standardize column names
df.columns = df.columns.str.lower()

df['some_numeric_column'] = df['some_numeric_column'].replace('[\$,]', '', regex=True).astype(float)

# Replace empty strings with NaN (if not already handled)
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Loop through columns and fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        # Fill categorical columns with mode
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col].fillna(mode[0], inplace=True)
    else:
        # Fill numeric columns with mean
        mean = df[col].mean(skipna=True)
        df[col].fillna(mean, inplace=True)

# Optional: Confirm there are no more missing values
print(df.isna().sum().sort_values(ascending=False).head())
