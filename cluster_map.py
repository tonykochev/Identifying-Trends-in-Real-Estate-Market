from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Initialize geolocator
geolocator = Nominatim(user_agent="city_locator")

# Function to geocode a single city/state
def get_lat_lon(city, state):
    try:
        location = geolocator.geocode(f"{city}, {state}, USA")
        if location:
            return pd.Series([location.latitude, location.longitude])
        else:
            return pd.Series([None, None])
    except GeocoderTimedOut:
        time.sleep(1)
        return get_lat_lon(city, state)  # Retry

# Get unique city/state combos
unique_locations = df[['city', 'state']].drop_duplicates()

# Geocode all
coords = unique_locations.apply(lambda row: get_lat_lon(row['city'], row['state']), axis=1)
coords.columns = ['lat', 'lon']
geo_df = pd.concat([unique_locations.reset_index(drop=True), coords], axis=1)

# Merge into city_stats
city_stats_with_geo = city_stats.merge(geo_df, on=['city', 'state'], how='left')

import plotly.express as px

# Run clustering again on normalized data
scaled = scaler.fit_transform(city_stats[['leasedsf', 'internal_class_rent',
                                          'availability_proportion', 'sublet_availability_proportion']])
kmeans = KMeans(n_clusters=3, random_state=42)
city_stats_with_geo['cluster'] = kmeans.fit_predict(scaled)

# Drop cities with missing geocode
map_df = city_stats_with_geo.dropna(subset=['lat', 'lon'])

# Plot
fig = px.scatter_geo(
    map_df,
    lat='lat',
    lon='lon',
    color='cluster',
    hover_name='city',
    scope='usa',
    title='Cluster Map of Cities Based on Corporate Leasing Metrics',
    template='plotly_white'
)
fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='black')))
fig.show()