from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=3,min_samples_leaf=10,min_samples_split=10,random_state=42,criterion='entropy')

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(accuracy_score(y_test,pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(model,filled=True)
plt.show()