# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Data Preparation
Load the dataset using pandas, clean column names, and separate input features (Size, Bedrooms) and target variables (Price, Occupants).
Step 2: Data Scaling
Apply feature scaling using StandardScaler to normalize the input data for better performance of the SGD model.
Step 3: Model Training
Create two SGDRegressor models and train them using the scaled features to predict Price and Occupants.
Step 4: Prediction
Take user input, scale it using the same scaler, and use the trained models to predict house price and number of occupants.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Srisanth R
RegisterNumber: 212225240156 
*/
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("house.csv")
#print(data.columns)
data.columns = data.columns.str.strip()
# Features (inputs)
X = data[['Size', 'Bedrooms']]

# Targets (outputs)
y_price = data['Price']
y_occ = data['Occupants']

# Scaling (important for SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

# Train models
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

# Input
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

# Scale input
new_data = scaler.transform([[size, bed]])

# Prediction
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:
<img width="1062" height="79" alt="Screenshot 2026-04-27 091146" src="https://github.com/user-attachments/assets/8fe8c986-ef2b-41f7-b7af-c8cbb4690923" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
