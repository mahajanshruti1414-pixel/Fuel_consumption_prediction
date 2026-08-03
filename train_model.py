import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. GENERATE / LOAD DATASET
# ==========================================
# For demonstration, using synthetic data with standard Auto MPG features.
# Replace this section with: df = pd.read_csv("auto_mpg.csv")
np.random.seed(42)
n_samples = 500

data = {
    'cylinders': np.random.choice([4, 6, 8], size=n_samples, p=[0.5, 0.3, 0.2]),
    'displacement': np.random.uniform(100, 450, size=n_samples),
    'horsepower': np.random.uniform(70, 230, size=n_samples),
    'weight': np.random.uniform(1800, 5000, size=n_samples),
    'acceleration': np.random.uniform(8, 25, size=n_samples),
}

# Formula-based fuel consumption target (L/100km) with added random noise
data['fuel_consumption'] = (
    0.02 * data['displacement'] + 
    0.015 * data['horsepower'] + 
    0.003 * data['weight'] - 
    0.1 * data['acceleration'] + 
    np.random.normal(0, 0.8, n_samples)
)

df = pd.DataFrame(data)

# ==========================================
# 2. FEATURE EXTRACTION & SPLITTING
# ==========================================
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df['fuel_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale continuous numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING & EVALUATION
# ==========================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics calculation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.3f} L/100km")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} L/100km")
print(f"R-squared Score (R²): {r2:.3f}")

# ==========================================
# 4. SAVING MODEL & SCALER
# ==========================================
joblib.dump(model, 'fuel_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\nSuccess: Saved 'fuel_model.joblib' and 'scaler.joblib' to disk!")