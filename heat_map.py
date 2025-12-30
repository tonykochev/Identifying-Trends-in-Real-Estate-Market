import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your lease data
df = pd.read_csv("filtered_leases.csv")  # Update with your file path

# --- Data Cleaning ---
df['leasedsf'] = pd.to_numeric(df['leasedsf'], errors='coerce')
df['overall_rent'] = pd.to_numeric(df['overall_rent'], errors='coerce')
df['city'] = df['city'].str.strip().str.title()
df['state'] = df['state'].str.upper()

# --- Aggregate by City and State ---
city_summary = df.groupby(['state', 'city']).agg({
    'leasedsf': 'sum',
    'overall_rent': 'mean',
    'building_id': 'count'
}).reset_index()

city_summary.rename(columns={
    'leasedsf': 'total_leased_sf',
    'overall_rent': 'avg_rent',
    'building_id': 'lease_activity'
}, inplace=True)

# --- Normalize data for heatmap ---
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
city_summary[['norm_leased_sf', 'norm_avg_rent', 'norm_lease_activity']] = scaler.fit_transform(
    city_summary[['total_leased_sf', 'avg_rent', 'lease_activity']]
)

# --- Create a "score" for leasing potential ---
city_summary['leasing_score'] = (city_summary['norm_leased_sf'] * 0.4 +
                                 city_summary['norm_avg_rent'] * -0.3 +  # Lower rent = better
                                 city_summary['norm_lease_activity'] * 0.3)

# --- Top Cities by Leasing Score ---
top_cities = city_summary.sort_values('leasing_score', ascending=False).head(20)

# --- Plot Heatmap ---
plt.figure(figsize=(12, 8))
heatmap_data = top_cities.pivot(index="city", columns="state", values="leasing_score")
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Top Leasing Locations by Score (Best = High Score)")
plt.xlabel("State")
plt.ylabel("City")
plt.tight_layout()
plt.show()
