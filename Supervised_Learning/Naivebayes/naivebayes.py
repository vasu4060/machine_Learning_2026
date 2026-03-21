from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt         


data = load_breast_cancer()
X = data.data
y = data.target

plt.scatter(X[:,0],X[:,1],y)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Breast Cancer Dataset")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)     

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   


model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)   


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))    



