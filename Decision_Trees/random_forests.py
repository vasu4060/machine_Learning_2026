from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=10,min_samples_leaf=5,min_samples_split=5,max_depth=10,oob_score=True,max_features='sqrt',random_state=42)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(accuracy_score(y_test,pred))