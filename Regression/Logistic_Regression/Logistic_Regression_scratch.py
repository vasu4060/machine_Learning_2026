import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()
X = data.data
y = data.target     
print("Feature names:", data.feature_names)
print("Target names:", data.target_names)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape) 


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state=42)


plt.figure(figsize=(8,5))
plt.scatter(X[:,1], y, c=y, cmap='viridis', alpha=0.6)
plt.title("Breast Cancer Dataset")      
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.grid(True)
plt.show()


model = LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)          
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

print("coefficients:",model.coef_)
print("intercept:",model.intercept_)    
print("scores:", model.score(X_test, Y_test))
      


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          


model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled,Y_train)
Y_pred = model.predict(X_test_scaled)      
print("After scaling")
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("scores:", model.score(X_test, Y_test))

print("coefficients:",model.coef_)
print("intercept:",model.intercept_)   
