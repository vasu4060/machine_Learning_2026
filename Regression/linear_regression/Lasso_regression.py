from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt         
import pandas as pd


data= fetch_california_housing()
x = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)       

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for alpha in alphas:
          print(f"Training Lasso Regression with alpha={alpha}")
          model = Lasso(alpha=alpha)
          model.fit(X_train,y_train)
          y_pred = model.predict(X_test)
          
          mse = mean_squared_error(y_test, y_pred)
          r2 = r2_score(y_test, y_pred)
          mae = mean_absolute_error(y_test, y_pred)
          evs = explained_variance_score(y_test, y_pred)
          
          print(f"Alpha: {alpha}")
          print("Mean Squared Error:", mse)
          print("R^2 Score:", r2)
          print("Mean Absolute Error:", mae)
          print("Explained Variance Score:", evs)
          print("Coefficients:", model.coef_)
          print("Intercept:", model.intercept_)
          print("-" * 30)