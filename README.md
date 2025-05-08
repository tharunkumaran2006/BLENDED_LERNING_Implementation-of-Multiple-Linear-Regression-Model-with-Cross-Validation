# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load and Prepare Data**  
   - Load the dataset using Pandas.  
   - Drop unnecessary columns like `car_ID` and `CarName`.  
   - Convert categorical variables into numerical format using one-hot encoding.

2. **Split the Data**  
   - Separate the dataset into features (`X`) and target variable (`y`).  
   - Split the dataset into training and testing sets using `train_test_split`.

3. **Build and Train the Model**  
   - Create a `LinearRegression` model instance.  
   - Fit the model on the training data.

4. **Evaluate the Model**  
   - Perform 5-fold cross-validation using `cross_val_score`.  
   - Evaluate the model on the test set using Mean Squared Error (MSE) and R² score.  
   - Visualize the actual vs predicted car prices using a scatter plot.

## Program:
### Program to implement the multiple linear regression model for predicting car prices with cross-validation.
### Developed by: THARUN V K
### RegisterNumber:  212223230231
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load and prepare data
# 1. Load and prepare data
url = "/content/CarPrice_Assignment.csv"
data = pd.read_csv(url)

# Simple preprocessing
data = data.drop(['car_ID', 'CarName'], axis=1)   # Remove unnecessary columns
data = pd.get_dummies(data, drop_first=True)      # Handle categorical variables

# 2. Split data
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate with cross-validation (simple version)
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Fold R² scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Average R²: {cv_scores.mean():.4f}")

# 5. Test set evaluation
y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 6. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
### EVALUATION WITH CROSS-VALIDATION:
![Screenshot 2025-04-23 193909](https://github.com/user-attachments/assets/06612350-deff-4bf1-99bd-b40a3f1ce6ec)

### TEST SET PERFORMANCE:
![Screenshot 2025-04-23 193918](https://github.com/user-attachments/assets/3daf1ea3-2562-4e2a-8bc7-b6206d77d2cb)

### ACTUAL VS PREDICTED PRICES:
![Screenshot 2025-04-23 193931](https://github.com/user-attachments/assets/28a956bb-b201-471a-84a8-3f0f3d012a00)

## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
