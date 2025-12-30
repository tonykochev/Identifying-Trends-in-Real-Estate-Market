import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv("filtered_leases.csv")  # Update this to your actual file path

# --- Basic Cleaning ---
# Standardize column names (lowercase, no spaces)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Convert dates and numerics
df['leasedsf'] = pd.to_numeric(df['leasedsf'], errors='coerce')
df['overall_rent'] = pd.to_numeric(df['overall_rent'], errors='coerce')
df['internal_class_rent'] = pd.to_numeric(df['internal_class_rent'], errors='coerce')
df['rba'] = pd.to_numeric(df['rba'], errors='coerce')
df['availability_proportion'] = pd.to_numeric(df['availability_proportion'], errors='coerce')
df['monthsigned'] = pd.to_numeric(df['monthsigned'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')

# Standardize text fields
df['city'] = df['city'].str.strip().str.title()
df['state'] = df['state'].str.upper()
df['internal_industry'] = df['internal_industry'].str.strip().str.title()
df['company_name'] = df['company_name'].str.strip().str.title()

# --- Drop rows with critical missing data ---
df.dropna(subset=['leasedsf', 'overall_rent', 'rba', 'availability_proportion'], inplace=True)

# --- Feature Engineering ---
df['rent_per_sf'] = df['overall_rent'] / df['leasedsf']
df['leasing_density'] = df['leasedsf'] / df['rba']
df['availability_score'] = 1 - df['availability_proportion']
df['year_month'] = df['year'].astype(str) + '-' + df['monthsigned'].astype(str).str.zfill(2)

# Optional: Create flags
df['is_sublet'] = df['transaction_type'].str.contains('sublet', case=False, na=False)
df['is_direct'] = df['transaction_type'].str.contains('direct', case=False, na=False)

# --- Normalize Key Metrics for Scoring ---
scaler = MinMaxScaler()
df[['norm_leasedsf', 'norm_rent', 'norm_density']] = scaler.fit_transform(
    df[['leasedsf', 'overall_rent', 'leasing_density']]
)

# Optional: Create a lease potential score (adjust weights as needed)
df['lease_score'] = (
    df['norm_leasedsf'] * 0.4 +
    df['norm_density'] * 0.3 -
    df['norm_rent'] * 0.3
)

# --- Save Cleaned Dataset (Optional) ---
df.to_csv("leases_cleaned.csv", index=False)
print("✅ Preprocessing complete. Cleaned data saved to 'leases_cleaned.csv'.")

# Preview the data
print(df.head())
