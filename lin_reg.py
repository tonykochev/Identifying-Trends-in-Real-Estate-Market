import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load and clean data
df = pd.read_csv('filtered_leases.csv')
df.columns = df.columns.str.lower()

# Replace empty strings with NaN
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Drop rows with missing target
df = df.dropna(subset=['leasedsf'])

# Define features and target
features = [
    'year', 'quarter', 'market', 'internal_submarket', 'internal_class',
    'internal_industry', 'space_type', 'cbd_suburban', 'rba',
    'internal_class_rent', 'overall_rent', 'available_space', 'availability_proportion'
]
target = 'leasedsf'

X = df[features]
y = df[target]

# Separate column types
cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(exclude='object').columns.tolist()

# Imputers
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')

# Preprocessing pipelines
cat_pipeline = Pipeline([
    ('imputer', cat_imputer),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('imputer', num_imputer)
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_features),
    ('num', num_pipeline, num_features)
])

# Final pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Optional: Score the model
from sklearn.metrics import mean_absolute_error, r2_score

print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

X['predicted_leasedsf'] = model.predict(X)

top_submarkets = X.groupby('internal_submarket')['predicted_leasedsf'].mean().sort_values(ascending=False)
top_cities = X.groupby('market')['predicted_leasedsf'].mean().sort_values(ascending=False)

print("Top Submarkets:\n", top_submarkets.head(10))
print("\nTop Cities:\n", top_cities.head(10))

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Plot Top 10 Cities
plt.figure(figsize=(12, 6))
sns.barplot(x=top_cities.head(10).values, y=top_cities.head(10).index, palette='Blues_d')
plt.title('Top 10 Cities by Predicted Leasing Performance')
plt.xlabel('Predicted Average Leased SF')
plt.ylabel('City')
plt.tight_layout()
plt.show()