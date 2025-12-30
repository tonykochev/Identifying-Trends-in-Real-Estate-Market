import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Load the data
df = pd.read_csv("filtered_leases.csv")

# Drop rows with missing key data
df = df.dropna(subset=[
    'city', 'leasedsf', 'internal_class_rent', 'overall_rent',
    'availability_proportion', 'sublet_availability_proportion'
])

# Group by city and calculate relevant stats
city_stats = df.groupby('city').agg({
    'leasedsf': 'sum',
    'internal_class_rent': 'mean',
    'overall_rent': 'mean',
    'availability_proportion': 'mean',
    'sublet_availability_proportion': 'mean',
    'building_id': 'count'  # use count of buildings as transaction volume
}).rename(columns={'building_id': 'transaction_count'}).reset_index()

# Normalize values for scoring
scaler = MinMaxScaler()
city_stats_scaled = city_stats.copy()
city_stats_scaled[['leasedsf', 'internal_class_rent', 'availability_proportion',
                   'sublet_availability_proportion']] = scaler.fit_transform(
    city_stats[['leasedsf', 'internal_class_rent', 'availability_proportion',
                'sublet_availability_proportion']]
)

# Score cities (higher is better)
city_stats_scaled['score'] = (
    (1 - city_stats_scaled['internal_class_rent']) * 0.35 +
    city_stats_scaled['leasedsf'] * 0.30 +
    city_stats_scaled['availability_proportion'] * 0.25 +
    (1 - city_stats_scaled['sublet_availability_proportion']) * 0.10
)

# Sort by score
top_cities = city_stats_scaled.sort_values(by='score', ascending=False)

# Merge with original values for output
top_cities_output = top_cities.merge(city_stats, on='city', suffixes=("_scaled", "_original"))

# Check the columns before further processing
print("Top Cities Output Columns Before Merge:", top_cities_output.columns)

# Load static city data (lat/lon)
cities = pd.read_csv("uscities.csv")

# Clean/merge just the necessary columns
cities = cities[['city', 'state_id', 'lat', 'lng']].drop_duplicates()

# Normalize casing to avoid merge mismatches
top_cities_output['city'] = top_cities_output['city'].str.lower().str.strip()
cities['city'] = cities['city'].str.lower().str.strip()

# Check the columns of cities DataFrame
print("Cities Data Columns:", cities.columns)

# Merge lat/lon into the `top_cities_output` DataFrame
city_summary = pd.merge(top_cities_output, cities, how='left', left_on=['city'], right_on=['city'])

# Check the columns of city_summary after merge
print("City Summary Columns After Merge:", city_summary.columns)

# Now `city_summary` has lat/lon columns you can use for mapping
# Create the map with Plotly
# Sort by score, ensuring the higher score cities are on top
city_summary_sorted = city_summary.sort_values(by="score", ascending=False)

fig = px.scatter_mapbox(
    city_summary_sorted,  # Use sorted data so higher scores are plotted last (on top)
    lat="lat",
    lon="lng",
    hover_name="city",
    hover_data={"state_id": True, "score": True},
    color="score",
    size="score",  # size markers by score
    color_continuous_scale="Viridis",
    zoom=3,
    height=600
)

# Adjust marker size and opacity for better visibility
fig.update_traces(marker=dict(
    sizemode='area',
    sizeref=2.*max(city_summary_sorted['score'])/(40.**2),  # Adjust size scaling factor if needed
    opacity=0.7  # Reduce opacity for better contrast
))

# Adjust opacity based on the score for a more pronounced effect
fig.update_traces(marker=dict(
    opacity=city_summary_sorted['score'] / max(city_summary_sorted['score'])  # Higher score = higher opacity
))

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title="Best US Cities for Corporate Leasing", margin={"r":0,"t":40,"l":0,"b":0})
fig.show()
