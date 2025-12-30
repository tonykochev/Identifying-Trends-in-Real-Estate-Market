import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the data
df = pd.read_csv("filtered_leases.csv")

# Drop rows with missing key data
df = df.dropna(subset=[
    'city', 'leasedsf', 'internal_class_rent', 'overall_rent',
    'availability_proportion', 'sublet_availability_proportion', 'internal_industry'
])

# Focus on business types: Tech, Legal, Financial
# One-hot encode the 'internal_industry' (business types)
df_encoded = pd.get_dummies(df, columns=['internal_industry'])

# Select the relevant features for clustering
features = [
    'leasedsf', 'internal_class_rent', 'overall_rent', 
    'availability_proportion', 'sublet_availability_proportion'
] + [col for col in df_encoded.columns if 'internal_industry' in col]

X = df_encoded[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to identify groups based on business types
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for Tech, Legal, and Financial industries
df_encoded['cluster'] = kmeans.fit_predict(X_scaled)

# Add the cluster information back to the city summary
city_summary = df_encoded.groupby(['city']).agg({
    'leasedsf': 'sum',
    'internal_class_rent': 'mean',
    'overall_rent': 'mean',
    'availability_proportion': 'mean',
    'sublet_availability_proportion': 'mean',
}).reset_index()

# Merge the cluster information with the city summary
city_summary['cluster'] = df_encoded.groupby('city')['cluster'].first().values

# Load static city data for lat/lon (use lat/lon from uscities.csv)
cities = pd.read_csv("uscities.csv")

# Clean/merge just the necessary columns
cities = cities[['city', 'state_id', 'lat', 'lng']].drop_duplicates()

# Normalize casing to avoid merge mismatches
city_summary['city'] = city_summary['city'].str.lower().str.strip()
cities['city'] = cities['city'].str.lower().str.strip()
cities['state_id'] = cities['state_id'].str.upper().str.strip()

# Merge lat/lon into your main data
city_summary = pd.merge(city_summary, cities, how='left', left_on=['city'], right_on=['city'])

# Map the clusters to business types
cluster_labels = {0: 'Tech', 1: 'Legal', 2: 'Financial'}

city_summary['industry_type'] = city_summary['cluster'].map(cluster_labels)

# Create a map to visualize clusters by city
fig = px.scatter_mapbox(
    city_summary,
    lat="lat",
    lon="lng",
    hover_name="city",
    hover_data=["state_id", "industry_type"],
    color="industry_type",
    color_discrete_map={"Tech": "blue", "Legal": "red", "Financial": "green"},
    title="Business Type Clusters for Corporate Leasing in US Cities",
    zoom=3,
    height=600
)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title="Tech, Legal, and Financial Business Clusters", margin={"r":0,"t":40,"l":0,"b":0})
fig.show()
