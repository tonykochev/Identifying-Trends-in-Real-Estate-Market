import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------
# 1. Load your cleaned data
# ----------------------------------------
df = pd.read_csv("leases_cleaned.csv")

# ----------------------------------------
# 2. Recompute calculated columns if needed
# ----------------------------------------
df["rent_per_sf"] = df["overall_rent"] / df["leasedsf"]
df["leasing_density"] = df["leasedsf"] / df["rba"]
df["availability_score"] = 1 - df["availability_proportion"]

# ----------------------------------------
# 3. Group by city & state (Base Summary)
# ----------------------------------------
city_summary = df.groupby(["state", "city"]).agg({
    "leasedsf": "sum",
    "rent_per_sf": "mean",
    "leasing_density": "mean",
    "availability_score": "mean",
    "company_name": "count"
}).reset_index()

# Rename columns for clarity
city_summary.rename(columns={
    "leasedsf": "total_leased_sf",
    "rent_per_sf": "avg_rent_per_sf",
    "leasing_density": "avg_density",
    "availability_score": "avg_availability_score",
    "company_name": "lease_activity"
}, inplace=True)

# 🆕 STEP 1: Filter out cities with low lease activity
city_summary = city_summary[city_summary["lease_activity"] >= 5]

# ----------------------------------------
# 4. Normalize features
# ----------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

city_summary[["norm_sf", "norm_rent", "norm_density", "norm_availability", "norm_activity"]] = scaler.fit_transform(
    city_summary[["total_leased_sf", "avg_rent_per_sf", "avg_density", "avg_availability_score", "lease_activity"]]
)

# ----------------------------------------
# 5. Compute lease score
# ----------------------------------------
city_summary["lease_score"] = (
    city_summary["norm_sf"] * 0.3 +
    city_summary["norm_density"] * 0.2 +
    city_summary["norm_availability"] * 0.1 +
    city_summary["norm_activity"] * 0.2 -
    city_summary["norm_rent"] * 0.2
)

# 🆕 OPTIONAL STEP 2: Group by industry
# Insert after city_summary if you want industry-specific views
industry_summary = df.groupby(["state", "city", "internal_industry"]).agg({
    "leasedsf": "sum",
    "rent_per_sf": "mean",
    "leasing_density": "mean",
    "availability_score": "mean",
    "company_name": "count"
}).reset_index()

# Rename and normalize
industry_summary.rename(columns={
    "leasedsf": "total_leased_sf",
    "rent_per_sf": "avg_rent_per_sf",
    "leasing_density": "avg_density",
    "availability_score": "avg_availability_score",
    "company_name": "lease_activity"
}, inplace=True)

industry_summary[["norm_sf", "norm_rent", "norm_density", "norm_availability", "norm_activity"]] = scaler.fit_transform(
    industry_summary[["total_leased_sf", "avg_rent_per_sf", "avg_density", "avg_availability_score", "lease_activity"]]
)

industry_summary["lease_score"] = (
    industry_summary["norm_sf"] * 0.3 +
    industry_summary["norm_density"] * 0.2 +
    industry_summary["norm_availability"] * 0.1 +
    industry_summary["norm_activity"] * 0.2 -
    industry_summary["norm_rent"] * 0.2
)

# 🆕 STEP 3: Sort and save
top_cities = city_summary.sort_values(by="lease_score", ascending=False)
top_cities.to_csv("top_leasing_cities.csv", index=False)

# 🆕 STEP 4: Visualize (Heatmap)
import seaborn as sns
import matplotlib.pyplot as plt

heatmap_data = top_cities.head(20).pivot(index="city", columns="state", values="lease_score")

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
plt.title("Top Cities for Leasing by Score")
plt.xlabel("State")
plt.ylabel("City")
plt.tight_layout()
plt.show()
