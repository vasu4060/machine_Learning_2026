from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt         
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,precision_score,recall_score,accuracy_score



data = load_iris()
X = data.data
y = data.target     


plt.scatter(X[:,0],X[:,1],y)
plt.xlabel(data.feature_names[0])
plt.ylabel("Target")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)     
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)       


c =  [0.01, 0.1, 1, 10]

for ci in c:
          print(f"Training Logistic Regression with C={ci}")
          model = LogisticRegression(C=ci, solver='lbfgs')
          model.fit(X_train_scaled, y_train)
          y_pred = model.predict(X_test_scaled)

          train_acc = model.score(X_train_scaled, y_train)
          test_acc = model.score(X_test_scaled, y_test)

          print("Train Accuracy:", train_acc)
          print("Test Accuracy:", test_acc)
          
          print("Classification Report:\n", classification_report(y_test, y_pred))
          print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
          print("Precision Score:", precision_score(y_test, y_pred, average='macro'))
          print("Recall Score:", recall_score(y_test, y_pred, average='macro'))
          print("Accuracy Score:", accuracy_score(y_test, y_pred))
          print("Model Coefficients:\n", model.coef_)
          print("Model Intercept:\n", model.intercept_)
          print("-" * 30)