from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


X, y = load_iris(return_X_y=True)


model = LogisticRegression(max_iter=1000)

scores = cross_val_score(model, X, y, cv=5)

print(scores)
print("Average accuracy:", scores.mean())




train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")
plt.legend()
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.show()