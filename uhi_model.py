import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: Your Real Satellite Data
# ============================================

data = {
    'City': ['Hyderabad', 'Chennai', 'Delhi', 'Nagpur', 'Ahmedabad', 'Jaipur', 'Lucknow'],
    'State': ['Telangana', 'Tamil Nadu', 'NCT', 'Maharashtra', 'Gujarat', 'Rajasthan', 'UP'],
    'LST': [40.90, 39.52, 39.40, 39.50, 44.00, 47.70, 30.20],
    'NDVI': [0.15, 0.14, 0.12, 0.15, 0.15, 0.12, 0.15],
    'NDBI': [-0.0017, -0.016, -0.01, -0.013, -0.008, 0.04, -0.02],
    'NDWI': [-0.18, -0.15, -0.15, -0.10, -0.18, 0.16, -0.18]
}

df = pd.DataFrame(data)
print("=== YOUR DATASET ===")
print(df)
print()

# ============================================
# STEP 2: Prepare Features and Target
# ============================================

X = df[['NDVI', 'NDBI', 'NDWI']]  # Input features
y = df['LST']                        # Target (temperature)

# Note: 7 cities is small — we'll use all data for training
# and demonstrate the model's learning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# STEP 3: Train Random Forest Model
# ============================================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("=== MODEL TRAINED SUCCESSFULLY ===")
print()

# ============================================
# STEP 4: Evaluate Model
# ============================================

y_pred = model.predict(X_test)
print("=== MODEL PERFORMANCE ===")
print(f"Predictions: {y_pred}")
print(f"Actual:      {y_test.values}")
print()

# Feature importance
importance = pd.DataFrame({
    'Feature': ['NDVI', 'NDBI', 'NDWI'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("=== WHICH FACTOR DRIVES HEAT THE MOST? ===")
print(importance)
print()

# ============================================
# STEP 5: Mitigation Simulator
# ============================================

print("=== MITIGATION SIMULATOR ===")
print("What if we increase green cover in Hyderabad?")
print()

# Current Hyderabad
current = pd.DataFrame({
    'NDVI': [0.15],
    'NDBI': [-0.0017],
    'NDWI': [-0.18]
})

# Scenario 1: Add more trees (increase NDVI by 0.1)
scenario1 = pd.DataFrame({
    'NDVI': [0.25],
    'NDBI': [-0.0017],
    'NDWI': [-0.18]
})

# Scenario 2: Add water bodies (increase NDWI by 0.05)
scenario2 = pd.DataFrame({
    'NDVI': [0.15],
    'NDBI': [-0.0017],
    'NDWI': [-0.13]
})

# Scenario 3: Reduce concrete (decrease NDBI by 0.05)
scenario3 = pd.DataFrame({
    'NDVI': [0.15],
    'NDBI': [-0.05],
    'NDWI': [-0.18]
})

current_temp = model.predict(current)[0]
temp1 = model.predict(scenario1)[0]
temp2 = model.predict(scenario2)[0]
temp3 = model.predict(scenario3)[0]

print(f"Current Hyderabad temp:              {current_temp:.2f}°C")
print(f"After adding trees (NDVI +0.1):      {temp1:.2f}°C  (change: {temp1-current_temp:.2f}°C)")
print(f"After adding water bodies (NDWI +0.05): {temp2:.2f}°C  (change: {temp2-current_temp:.2f}°C)")
print(f"After reducing concrete (NDBI -0.05): {temp3:.2f}°C  (change: {temp3-current_temp:.2f}°C)")

# ============================================
# STEP 6: Visualizations
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Urban Heat Island Analysis — BAH 2026', fontsize=14, fontweight='bold')

# Plot 1: City temperatures
axes[0].bar(df['City'], df['LST'], color=['red','orange','orange','orange','red','darkred','green'])
axes[0].set_title('Land Surface Temperature by City')
axes[0].set_ylabel('LST (°C)')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Feature importance
axes[1].bar(importance['Feature'], importance['Importance'],
            color=['green', 'gray', 'blue'])
axes[1].set_title('What Drives Urban Heat the Most?')
axes[1].set_ylabel('Importance Score')

# Plot 3: Mitigation scenarios for Hyderabad
scenarios = ['Current', '+Trees', '+Water', '-Concrete']
temps = [current_temp, temp1, temp2, temp3]
colors = ['red', 'green', 'blue', 'orange']
axes[2].bar(scenarios, temps, color=colors)
axes[2].set_title('Hyderabad — Mitigation Scenarios')
axes[2].set_ylabel('Predicted LST (°C)')
axes[2].set_ylim(min(temps)-2, max(temps)+2)

plt.tight_layout()
plt.savefig('uhi_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print("=== DONE! Chart saved as uhi_analysis.png ===")