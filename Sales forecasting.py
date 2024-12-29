# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic sales data
# Replace this with your actual sales dataset
data = {
    "Month": range(1, 13),  # Months 1 to 12
    "Sales": [200, 220, 250, 270, 300, 310, 400, 420, 450, 460, 480, 500]
}
df = pd.DataFrame(data)

# Feature and target variable
X = df[["Month"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the results
plt.scatter(X, y, color="blue", label="Actual Sales")
plt.plot(X, model.predict(X), color="red", label="Forecasted Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()

# Forecast sales for future months
future_months = pd.DataFrame({"Month": [13, 14, 15]})
future_sales = model.predict(future_months)
print("Forecasted Sales for Future Months:")
print(future_months.assign(Forecasted_Sales=future_sales))