from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix





data = load_iris()
X = data.data
y = data.target
print("Feature names:", data.feature_names)
print("Target names:", data.target_names)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
print(LogisticRegression)
model = LogisticRegression(solver='lbfgs')
print(LogisticRegression)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

print("Model Coefficients:\n", model.coef_)
print("Model Intercept:\n", model.intercept_)

print("Shape of coefficient: ", model.coef_.shape)
print("intercept shape:", model.intercept_.shape)


