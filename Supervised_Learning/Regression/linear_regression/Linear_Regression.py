import pandas as pd
import yfinance as yf

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error


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

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)    

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")



