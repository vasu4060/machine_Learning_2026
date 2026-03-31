from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = load_breast_cancer()
X = data.data
y = data.target     

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Breast Cancer Dataset")
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=10, random_state=42)
model.fit(X_train_scaled, Y_train)
y_pred = model.predict(X_test_scaled)          

print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))        
print("Accuracy Score:", accuracy_score(Y_test, y_pred))    
