
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

data =load_iris()
X = data.data
y= data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = np.c_[np.ones((X_train_scaled.shape[0],1)), X_train_scaled]
X_test  = np.c_[np.ones((X_test_scaled.shape[0],1)), X_test_scaled]


num_classes = 3

Y_train = np.eye(num_classes)[y_train]
Y_test  = np.eye(num_classes)[y_test]

n_features = X_train_scaled.shape[1]

W = np.random.randn(n_features, num_classes) * 0.01

def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(Y_true, Y_pred):

    m = Y_true.shape[0]

    return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / m



learning_rate = 0.1
iterations = 2000

m = X_train.shape[0]

for i in range(iterations):

    Z = X_train_scaled @ W

    Y_pred = softmax(Z)

    loss = cross_entropy(Y_train, Y_pred)

    grad = (1/m) * X_train_scaled.T @ (Y_pred - Y_train)

    W -= learning_rate * grad

    if i % 200 == 0:
        print("Iteration:", i, "Loss:", loss)

def predict(X, W):

    Z = X @ W
    probs = softmax(Z)

    return np.argmax(probs, axis=1)

y_train_pred = predict(X_train_scaled, W)
y_test_pred = predict(X_test_scaled, W)

train_acc = np.mean(y_train_pred == y_train)
test_acc = np.mean(y_test_pred == y_test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)