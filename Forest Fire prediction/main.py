import random
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Set up date range
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-12-31')
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
# Create empty data frame
weather_data = pd.DataFrame({'date': date_range})
# Add temperature data
temp_mean = 70 # Mean temperature
temp_std = 10 # Standard deviation
weather_data['temperature'] = [random.normalvariate(temp_mean, temp_std) for i in range(len(date_range))]
# Add humidity data
humid_mean = 50 # Mean humidity
humid_std = 15 # Standard deviation
weather_data['humidity'] = [random.normalvariate(humid_mean, humid_std) for i in range(len(date_range))]
# Add wind speed data
wind_mean = 5 # Mean wind speed in miles per hour
wind_std = 2 # Standard deviation
weather_data['wind_speed'] = [random.normalvariate(wind_mean, wind_std) for i in range(len(date_range))]
# Add fuel moisture data
fuel_moist_mean = 10 # Mean fuel moisture
fuel_moist_std = 3 # Standard deviation
weather_data['fuel_moisture'] = [random.normalvariate(fuel_moist_mean, fuel_moist_std) for i in range(len(date_range))]
# Add topography data
topo_mean = 500 # Mean elevation in feet
topo_std = 100 # Standard deviation
weather_data['elevation'] = [random.normalvariate(topo_mean, topo_std) for i in range(len(date_range))]
# Save data to CSV file
weather_data.to_csv('fictional_weather_data.csv', index=False)
# Load weather data from CSV file
weather_data = pd.read_csv('fictional_weather_data.csv')
# Check for missing values
print(weather_data.isnull().sum())
# Remove missing values
weather_data.dropna(inplace=True)
# Remove outliers using z-score
z_scores = np.abs((weather_data - weather_data.mean()) / weather_data.std())
weather_data = weather_data[(z_scores < 3).all(axis=1)]
# Save cleaned data to CSV file
weather_data.to_csv('cleaned_weather_data.csv', index=False)
# Load historical wildfire data from CSV file
wildfire_data = pd.read_csv('historical_wildfire_data.csv')
# Create a new column for wildfire occurrence
wildfire_data['wildfire_occurrence'] = [1 if area > 0 else 0 for area in wildfire_data['burned_area']]
# Save data to CSV file
wildfire_data.to_csv('wildfire_occurrence_data.csv', index=False)
# Load data
data = pd.read_csv('wildfire_occurrence_data.csv')
# Define input and output variables
X = data.drop('wildfire_occurrence', axis=1)
y = data['wildfire_occurrence']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the machine learning model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the machine learning model
rf.fit(X_train, y_train)
# Make predictions on the test set
y_pred = rf.predict(X_test)
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
# Load preprocessed weather and topography data
weather_data = pd.read_csv('cleaned_weather_data.csv')
topography_data = gpd.read_file('topography_data.geojson')
# Merge weather and topography data on the common column 'id'
merged_data = pd.merge(weather_data, topography_data, on='elevation')
# Load historical wildfire occurrence data
wildfire_data = pd.read_csv('wildfire_occurrence_data.csv')
# Merge merged data and wildfire occurrence data on the common column 'date'
merged_data['date'] = pd.to_datetime(merged_data['date'])
wildfire_data['date'] = pd.to_datetime(wildfire_data['date'])
final_data = pd.merge(merged_data, wildfire_data, on='date')
# Split the data into training and testing sets
train_data = final_data[final_data['year'] <= 2017]
test_data = final_data[final_data['year'] > 2017]
# Define the features to be used in the machine learning model
features = ['temperature', 'humidity', 'wind_speed', 'fuel_moisture', 'elevation']
# Train a random forest classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(train_data[features], train_data['wildfire_occurrence'])
# Predict wildfire occurrence probabilities for the test data
test_data_copy = test_data.copy()
test_data_copy.loc[:,'wildfire_probability'] = rfc.predict_proba(test_data_copy[features])[:,1]
# Map the predicted wildfire occurrence probabilities
fig, ax = plt.subplots(figsize=(10,10))
topography_data.plot(ax=ax, facecolor='white', edgecolor='black')
test_data.plot(ax=ax, column='wildfire_probability', cmap='Reds', legend=True, legend_kwds={'label': "Probability of Wildfire Occurrence"})
plt.title("Predicted Wildfire Occurrence Probabilities")
plt.show()