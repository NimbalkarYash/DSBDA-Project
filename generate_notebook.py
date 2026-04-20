import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split('\n')]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split('\n')]
    })

add_code("""# Install required libraries in the current Jupyter kernel
%pip install pandas numpy matplotlib seaborn scikit-learn
""")

add_markdown("# Advanced Air Pollution Analysis & City-Level ML Models")

add_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

import warnings
warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)
""")

add_markdown("## 1. Robust Data Processing & Imputation\nHandling inconsistent columns from datasets 2020 to 2023.")

add_code("""
files = {
    2020: "dataset---2020.csv",
    2021: "dataset---2021.csv",
    2022: "dataset---2022.csv",
    2023: "dataset---2023.csv"
}

dfs = []

def clean_column_names(cols):
    cleaned = []
    for col in cols:
        c = re.sub(r'[^a-zA-Z0-9]', '', str(col).lower())
        if 'state' in c:
            cleaned.append('State')
        elif 'city' in c or 'town' in c:
            cleaned.append('City')
        elif 'so2' in c:
            cleaned.append('SO2')
        elif 'no2' in c:
            cleaned.append('NO2')
        elif 'pm10' in c:
            cleaned.append('PM10')
        elif 'pm25' in c:
            cleaned.append('PM25')
        elif 'year' in c:
            cleaned.append('Year_Drop')
        else:
            cleaned.append(col)
    return cleaned

for year, file in files.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        df.columns = clean_column_names(df.columns)
        if 'Year_Drop' in df.columns:
            df.drop('Year_Drop', axis=1, inplace=True)
        df['Year'] = year
        
        target_cols = ['Year', 'State', 'City', 'SO2', 'NO2', 'PM10', 'PM25']
        
        for col in target_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        dfs.append(df[target_cols])

final_df = pd.concat(dfs, ignore_index=True)

# Clean null variants
final_df.replace(['NM', '-', 'NA', ' ', ''], np.nan, inplace=True)

pollutants = ['SO2', 'NO2', 'PM10', 'PM25']
for p in pollutants:
    final_df[p] = pd.to_numeric(final_df[p], errors='coerce')

final_df['City'] = final_df['City'].str.strip().str.title()
final_df['State'] = final_df['State'].str.strip().str.title()

# Imputation (City -> State -> Global)
for col in pollutants:
    final_df[col] = final_df.groupby('City')[col].transform(lambda x: x.fillna(x.mean()))
for col in pollutants:
    final_df[col] = final_df.groupby('State')[col].transform(lambda x: x.fillna(x.mean()))
for col in pollutants:
    final_df[col] = final_df[col].fillna(final_df[col].mean())

# Missing City or State rows drop
final_df.dropna(subset=['City', 'State'], inplace=True)

final_df.head()
""")

add_markdown("## 2. Engineered Features")

add_code("""
# 1. Total Pollution Proxy (Simple sum for Top 10 lists)
final_df['Total'] = final_df[['SO2', 'NO2', 'PM10', 'PM25']].sum(axis=1)

# 2. AQI Classification Category
def categorize_aqi(pm25):
    if pm25 <= 30: return 'Good'
    if pm25 <= 60: return 'Satisfactory'
    if pm25 <= 90: return 'Moderate'
    if pm25 <= 120: return 'Poor'
    return 'Severe'

final_df['Air_Quality_Category'] = final_df['PM25'].apply(categorize_aqi)

final_df[['City', 'Year', 'PM25', 'Total', 'Air_Quality_Category']].head()
""")

add_markdown("## 3. Original Exploratory Data Analysis & Visualizations")

add_code("""
# Histograms
final_df[pollutants].hist(figsize=(10,8), color='teal', edgecolor='black')
plt.suptitle('Distribution of Pollutants', fontsize=16)
plt.show()
""")

add_code("""
# Scatterplot NO2 vs PM2.5
plt.figure(figsize=(8,6))
plt.scatter(final_df['NO2'], final_df['PM25'], alpha=0.5, color='darkred')
plt.xlabel("NO2")
plt.ylabel("PM2.5")
plt.title("Scatter Analysis: NO2 vs PM2.5")
plt.grid(True)
plt.show()
""")

add_code("""
# Boxplot & Violinplot
plt.figure(figsize=(10,5))
final_df[pollutants].plot.box(title="Boxplot Check for Outliers", grid=True)
plt.show()

plt.figure(figsize=(10,5))
sns.violinplot(data=final_df[pollutants], palette='Set2')
plt.title('Violin Plot showing Pollutant Kernel Density')
plt.show()
""")

add_code("""
# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(final_df[pollutants].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Pollutants')
plt.show()
""")

add_code("""
# Trends over the years
plt.figure(figsize=(10,5))
final_df.groupby('Year')[pollutants].mean().plot(marker='o', linewidth=2)
plt.title("Yearly Average Pollution Trend (National)")
plt.ylabel("Levels")
plt.grid(True)
plt.legend(title='Pollutant')
plt.show()
""")

add_markdown("## 4. Ranking Data (Top & Least Polluted)")

add_code("""
# Aggregate ranking data
city_pollution = final_df.groupby('City')[pollutants + ['Total']].mean()
state_pollution = final_df.groupby('State')[pollutants + ['Total']].mean()

top_cities = city_pollution.sort_values(by='Total', ascending=False).head(10)
least_cities = city_pollution.sort_values(by='Total', ascending=True).head(10)

top_states = state_pollution.sort_values(by='Total', ascending=False).head(10)
least_states = state_pollution.sort_values(by='Total', ascending=True).head(10)

print("--- TOP 10 MOST POLLUTED CITIES ---")
display(top_cities[['Total']])

print("\\n--- TOP 10 LEAST POLLUTED CITIES ---")
display(least_cities[['Total']])

print("\\n--- TOP 10 MOST POLLUTED STATES ---")
display(top_states[['Total']])

print("\\n--- TOP 10 LEAST POLLUTED STATES ---")
display(least_states[['Total']])
""")

add_markdown("## 5. City Improvement / Worsening Tool (2020 vs 2023 Analysis)")

add_code("""
# Compare strictly the differential of 2023 vs 2020 baseline
city_year = final_df.groupby(['City', 'Year'], as_index=False)[pollutants].mean()

data_2020 = city_year[city_year['Year'] == 2020].copy()
data_2023 = city_year[city_year['Year'] == 2023].copy()

merged = pd.merge(data_2020, data_2023, on='City', how='inner', suffixes=('_2020', '_2023'))

for col in pollutants:
    merged[col + '_change'] = merged[col + '_2023'] - merged[col + '_2020']

change_cols = [col + '_change' for col in pollutants]
merged['Total Change'] = merged[change_cols].sum(axis=1)

improved = merged.sort_values(by='Total Change', ascending=True).head(10)
worsened = merged.sort_values(by='Total Change', ascending=False).head(10)

print("--- TOP 10 CITIES THAT IMPROVED MOST (Values Decreased) ---")
display(improved[['City', 'Total Change']])

print("\\n--- TOP 10 CITIES THAT WORSENED MOST (Values Increased) ---")
display(worsened[['City', 'Total Change']])
""")

add_markdown("## 6. Original Linear Regression (National Year Prediction)")

add_code("""
from sklearn.linear_model import LinearRegression

grouped = final_df.groupby('Year')[pollutants].mean()
X_lin = grouped.index.values.reshape(-1, 1)

models_lin = {}
pred_2024 = {}

for col in pollutants:
    model = LinearRegression()
    model.fit(X_lin, grouped[col])
    models_lin[col] = model
    pred_2024[col] = model.predict(np.array([[2024]]))[0]

print("Global Linear Regression Prediction for 2024:\\n", pred_2024)

# Visualization
years_ext = list(X_lin.flatten()) + [2024]

plt.figure(figsize=(10,6))
for col in pollutants:
    values = list(grouped[col]) + [pred_2024[col]]
    plt.plot(years_ext, values, marker='o', label=col, linestyle='dashed')

plt.title("National Pollution Trend Extended to 2024")
plt.xlabel("Year")
plt.legend()
plt.grid(True)
plt.show()
""")

add_markdown("## 7. Advanced Machine Learning: Regressor & Classifier")

add_code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Prepare categorical data
ml_df = final_df.copy()

le_city = LabelEncoder()
le_state = LabelEncoder()
le_target = LabelEncoder()

ml_df['City_Code'] = le_city.fit_transform(ml_df['City'])
ml_df['State_Code'] = le_state.fit_transform(ml_df['State'])
ml_df['Target_Category_Code'] = le_target.fit_transform(ml_df['Air_Quality_Category'])

# Shared ML Features
X_rf = ml_df[['Year', 'State_Code', 'City_Code']]

# REGRESSION TARGET
y_reg = ml_df['PM25']

# CLASSIFICATION TARGET
y_cls = ml_df['Target_Category_Code']

# Splits
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_rf, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_rf, y_cls, test_size=0.2, random_state=42)

print("Preparing ML Pipelines...")
""")

add_code("""
# -- 1. Random Forest Regressor (Predicting exact PM2.5 digits) --
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
preds_reg = rf_reg.predict(X_test_reg)

print("=== REGRESSION RESULTS (PM2.5 prediction) ===")
print(f"R-Squared Score: {r2_score(y_test_reg, preds_reg):.3f}")
print(f"MSE: {mean_squared_error(y_test_reg, preds_reg):.2f}\\n")

# -- 2. Random Forest Classifier (Predicting Air Quality Category) --
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_train_cls, y_train_cls)
preds_cls = rf_cls.predict(X_test_cls)

print("=== CLASSIFICATION RESULTS (Air Quality Good/Severe/etc) ===")
print(classification_report(y_test_cls, preds_cls, target_names=le_target.classes_))
""")

add_code("""
# Feature importance plot mapping why models make decisions
importances = rf_reg.feature_importances_

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=X_rf.columns, palette='mako')
plt.title('Feature Importances in ML Models (What Drives Pollution)')
plt.xlabel('Importance Weight')
plt.show()
""")

with open('Final_Pollution_Project.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Final_Pollution_Project.ipynb rebuilt with ALL requested features!")
