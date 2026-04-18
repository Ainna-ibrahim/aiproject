import pandas as pd

# Load dataset
data = pd.read_csv("data/student-mat.csv")

# Features (input)
X = data.drop("G3", axis=1)

# Target (output)
y = data["G3"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
# Convert categorical data to numeric
X = pd.get_dummies(X)

print("After encoding:", X.shape)
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully 🚀")
# Make predictions
y_pred = model.predict(X_test)

# Show first 5 predictions
print("Predictions:", y_pred[:5])

# Show actual values
print("Actual:", y_test.values[:5])
from sklearn.metrics import mean_absolute_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted")
plt.show()
print("Model Used: Random Forest")