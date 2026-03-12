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



X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          

Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)  


import numpy as np

X_train = np.c_[np.ones((X_train_scaled.shape[0],1)), X_train_scaled]
X_test  = np.c_[np.ones((X_test_scaled.shape[0],1)), X_test_scaled]

np.random.seed(42)
theta = np.random.randn(X_train.shape[1],1) * 0.01
print(theta)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, theta):
    m = len(y)
    z = X @ theta
    y_hat = sigmoid(z)
    
    epsilon = 1e-8
    loss = -(1/m) * np.sum(
        y * np.log(y_hat + epsilon) +
        (1-y) * np.log(1 - y_hat + epsilon)
    )
    return loss


def compute_loss_ridge(X, y, theta, lambda_):
    m = len(y)
    z = X @ theta
    y_hat = sigmoid(z)
    
    epsilon = 1e-8
    loss_ridge = -(1/m) * np.sum(
        y * np.log(y_hat + epsilon) +
        (1-y) * np.log(1 - y_hat + epsilon)
    ) + (lambda_ / (2*m)) * np.sum(theta[1:] ** 2)
    
    return loss_ridge

def compute_loss_lasso(X, y, theta, lambda_):
    m = len(y)
    z = X @ theta
    y_hat = sigmoid(z)
    
    epsilon = 1e-8
    loss_lasso = -(1/m) * np.sum(
        y * np.log(y_hat + epsilon) +
        (1-y) * np.log(1 - y_hat + epsilon)
    ) + (lambda_ / m) * np.sum(np.abs(theta[1:]))
    
    return loss_lasso   

def compute_loss_elasticnet(X, y, theta, lambda_, alpha):
    m = len(y)
    z = X @ theta
    y_hat = sigmoid(z)
    
    epsilon = 1e-8
    loss_elasticnet = -(1/m) * np.sum(
        y * np.log(y_hat + epsilon) +
        (1-y) * np.log(1 - y_hat + epsilon)
    ) + (lambda_ / (2*m)) * (alpha * np.sum(theta[1:] ** 2) + (1-alpha) * np.sum(np.abs(theta[1:])))
    
    return loss_elasticnet



def compute_gradient(X, y, theta):
    m = len(y)
    y_hat = sigmoid(X @ theta)
    grad = (1/m) * (X.T @ (y_hat - y))
    return grad     
def compute_gradient_ridge(X, y, theta, lambda_):
    m = len(y)
    y_hat = sigmoid(X @ theta)
    grad_ridge = (1/m) * (X.T @ (y_hat - y)) + (lambda_ / m) * np.r_[[[0]], theta[1:]]
    return grad_ridge
def compute_gradient_lasso(X, y, theta, lambda_): 
    m = len(y)
    y_hat = sigmoid(X @ theta)
    grad_lasso = (1/m) * (X.T @ (y_hat - y)) + (lambda_ / m) * np.r_[[[0]], np.sign(theta[1:])]
    return grad_lasso
def compute_gradient_elasticnet(X, y, theta, lambda_, alpha):
    m = len(y)
    y_hat = sigmoid(X @ theta)
    grad_elasticnet = (1/m) * (X.T @ (y_hat - y)) + (lambda_ / m) * np.r_[[[0]], alpha * theta[1:] + (1-alpha) * np.sign(theta[1:])]
    return grad_elasticnet  




learning_rate = 0.01
iterations = 3000

for i in range(iterations):
    grad = compute_gradient(X_train, Y_train, theta)
    theta -= learning_rate * grad
    grad_ridge = compute_gradient_ridge(X_train, Y_train, theta, lambda_=1.0)
    theta -= learning_rate * grad_ridge
    grad_lasso = compute_gradient_lasso(X_train, Y_train, theta, lambda_=0.1)
    theta -= learning_rate * grad_lasso
    grad_elasticnet = compute_gradient_elasticnet(X_train, Y_train, theta, lambda_=1.0, alpha=0.5)
    theta -= learning_rate * grad_elasticnet
    
    if i % 200 == 0:
        print("Iteration:", i, "Loss:", compute_loss(X_train, Y_train, theta))
        print("Iteration:", i, "Ridge Loss:", compute_loss_ridge(X_train, Y_train, theta, lambda_=1.0))
        print("Iteration:", i, "Lasso Loss:", compute_loss_lasso(X_train, Y_train, theta, lambda_=0.01))
        print("Iteration:", i, "ElasticNet Loss:", compute_loss_elasticnet(X_train, Y_train, theta, lambda_=1.0, alpha=0.5))    



def predict(X, theta):
    probs = sigmoid(X @ theta)
    return (probs >= 0.5).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


train_acc = accuracy(Y_train, predict(X_train, theta))
test_acc = accuracy(Y_test, predict(X_test, theta))

print(f"y intercept and Coefficient of theta are : y - intercept :{theta[0][0]}, other ceofficients :{theta[1:].flatten()}")


print(f"Custom regression train accuracy:{train_acc} and Test accuracy:{test_acc}")