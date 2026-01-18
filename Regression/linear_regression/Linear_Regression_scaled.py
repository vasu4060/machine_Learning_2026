import pandas as pd
import yfinance as yf

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.metrics import r2_score,mean_squared_error

import matplotlib.pyplot as plt


df = yf.download("GOOGL", start="2010-01-01")
print(df.head())

df = yf.download("GOOGL", period="5y")
print(df.columns)
df["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100
df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100
df =df[['Close','Open','Volume','HL_PCT','PCT_change']]
print(df.head(10))


forecast_col = 'Close'
df.fillna(-99999,inplace=True)
df['label'] = df[forecast_col].shift(-1)

print(df.head(10))     



print(df.tail(10))  
df.dropna(inplace=True)
print(df.tail(10))

X = df.drop('label',axis=1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

models= {
          "Linear" : LinearRegression(),
          "Ridge" : Ridge(alpha=1.0),
          "Lasso" : Lasso(alpha=0.01)


} 

results = {}

for name, model in models.items():
          model.fit(X_train_scaled,y_train)

          y_pred_scaled = model.predict(X_test_scaled)

          rmse_scaled = np.sqrt(mean_squared_error(y_test,y_pred_scaled))

          r2_scaled = r2_score(y_test,y_pred_scaled)

          print(name)

          print(f"RMSE: {rmse_scaled:.2f}")
          print(f"R² Score: {r2_scaled:.4f}")
          results[name] = {"RMSE":rmse_scaled,"r2":r2_scaled}



for name,metrics in results.items():
         print(f"{model}: RMSE={metrics['RMSE']:.2f}, R2={metrics['r2']:.4f}")



plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred_scaled, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.title("Actual vs Predicted Closing Price (Test Set)")
plt.legend()
plt.show()