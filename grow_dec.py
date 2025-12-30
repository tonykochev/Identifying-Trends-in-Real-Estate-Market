import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('filtered_leases.csv')

# Group by city and year, count number of leases
leases_by_city_year = df.groupby(['city', 'year']).size().reset_index(name='lease_count')

# Pivot to make years into columns (optional but useful for comparison)
pivot_table = leases_by_city_year.pivot(index='city', columns='year', values='lease_count').fillna(0)

# Calculate year-over-year growth — from the last two available years
if pivot_table.shape[1] >= 2:
    years = sorted(pivot_table.columns)
    pivot_table['growth_rate'] = pivot_table[years[-1]] - pivot_table[years[-2]]
else:
    print("Not enough years of data to calculate growth.")
    pivot_table['growth_rate'] = 0

# Sort by growth
growing_cities = pivot_table.sort_values(by='growth_rate', ascending=False)
declining_cities = pivot_table.sort_values(by='growth_rate')

# Show top 10 growing and declining cities
print("Top 10 Growing Cities:\n", growing_cities.head(10))
print("\nTop 10 Declining Cities:\n", declining_cities.head(10))

# Optionally save to CSV
growing_cities.to_csv('city_growth_trends.csv')

# Load the data (or reuse your existing pivot_table)
pivot_table = pd.read_csv('city_growth_trends.csv', index_col='city')

# Sort by growth rate
pivot_sorted = pivot_table.sort_values(by='growth_rate', ascending=False)

# Pivot to make years into columns (optional for plotting)
# But for a line plot, we’ll stick with the tidy format
plt.figure(figsize=(14, 7))

# Plot a line for each city (top 5 by total leases to avoid clutter)
top_cities = (
    leases_by_city_year.groupby('city')['lease_count']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
)

# Filter data for top cities
top_city_data = leases_by_city_year[leases_by_city_year['city'].isin(top_cities)]

# Plot
sns.lineplot(data=top_city_data, x='year', y='lease_count', hue='city', marker='o')
plt.title('Yearly Leasing Trends by Top Cities')
plt.xlabel('Year')
plt.ylabel('Number of Leases')
plt.legend(title='City')
plt.grid(True)
plt.tight_layout()
plt.show()
