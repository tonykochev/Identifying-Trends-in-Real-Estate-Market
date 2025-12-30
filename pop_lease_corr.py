import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSVs
lease_df = pd.read_csv("filtered_leases.csv")
pop_df = pd.read_csv("sub-est2023.csv", encoding="ISO-8859-1")

# Mapping from state full names to abbreviations
state_abbrev = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc'
}

# --------------------------
# Step 1: Clean and prepare lease data
# --------------------------
lease_df = lease_df[['city', 'state', 'leasedsf', 'overall_rent', 'availability_proportion']]
#lease_df.dropna(subset=['city', 'state', 'leasedsf', 'overall_rent', 'availability_proportion'], inplace=True)
lease_df['city'] = lease_df['city'].str.strip().str.lower()
lease_df['state'] = lease_df['state'].str.strip().str.lower()

lease_agg = lease_df.groupby(['city', 'state']).agg({
    'leasedsf': 'sum',
    'overall_rent': 'mean',
    'availability_proportion': 'mean'
}).reset_index()

# --------------------------
# Step 2: Clean and prepare population data
# --------------------------
pop_df = pop_df[['NAME', 'STNAME', 'POPESTIMATE2020', 'POPESTIMATE2023']].copy()
pop_df.dropna(inplace=True)

# Create abbreviated state first, then rename columns
pop_df['state'] = pop_df['STNAME'].str.strip().str.lower().map(state_abbrev)
pop_df['city'] = pop_df['NAME'].str.strip().str.lower()

# Drop rows where mapping failed
pop_df.dropna(subset=['state'], inplace=True)

# Calculate growth rate
pop_df['pop_growth_rate'] = (pop_df['POPESTIMATE2023'] - pop_df['POPESTIMATE2020']) / pop_df['POPESTIMATE2020']

# --------------------------
# Step 3: Merge and Debug
# --------------------------
# Debug outer merge
debug_merge = pd.merge(lease_agg, pop_df, on=['city', 'state'], how='outer', indicator=True)
print("Merge result counts:\n", debug_merge['_merge'].value_counts())

# Now filter to only inner joins
merged_df = debug_merge[debug_merge['_merge'] == 'both'].copy()

print("Merged shape:", merged_df.shape)

# --------------------------
# Step 4: Correlation analysis
# --------------------------
correlation = merged_df[['pop_growth_rate', 'leasedsf', 'overall_rent', 'availability_proportion']].corr()
print("Correlation Matrix:\n", correlation)

# --------------------------
# Step 5: Scoring system
# --------------------------
score_df = merged_df.copy()

scaler = MinMaxScaler()
score_df[['pop_growth_rate', 'leasedsf']] = scaler.fit_transform(score_df[['pop_growth_rate', 'leasedsf']])
score_df['inv_availability'] = 1 - scaler.fit_transform(score_df[['availability_proportion']])

score_df['score'] = (
    0.4 * score_df['pop_growth_rate'] +
    0.4 * score_df['leasedsf'] +
    0.2 * score_df['inv_availability']
)

top_cities = score_df.sort_values(by='score', ascending=False)[['city', 'state', 'score']].head(20)
print("\nTop 20 Cities for Leasing Based on Growth and Demand:\n", top_cities)

# Optional save
score_df.to_csv("city_leasing_scores.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a new column with formatted city/state for display
top_cities['city_state'] = top_cities['city'].str.title() + ", " + top_cities['state'].str.upper()

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Barplot
barplot = sns.barplot(
    data=top_cities,
    y='city_state',
    x='score',
    palette='viridis'
)

# Add value labels
for index, row in top_cities.iterrows():
    barplot.text(row['score'] + 0.005, index, f"{row['score']:.2f}", va='center')

# Labels and title
plt.title("Top 20 U.S. Cities for Business Leasing (2020–2023)", fontsize=16)
plt.xlabel("Composite Score (Population Growth & Leasing Demand)", fontsize=12)
plt.ylabel("City", fontsize=12)
plt.tight_layout()

# Show plot
plt.show()

