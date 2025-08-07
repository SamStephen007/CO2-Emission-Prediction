import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('CO2 Emissions_Canada.csv')

# Select features and target
X = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)',
        'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']]
y = df['CO2 Emissions(g/km)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ElasticNet model
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model and scaler
pickle.dump(model, open('model/co2_model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
