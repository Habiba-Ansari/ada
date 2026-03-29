# Import libraries
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
data = pd.read_csv('data.csv')
ts = data['value']

# Split into train & test
train = ts[:int(len(ts)*0.8)]
test = ts[int(len(ts)*0.8):]

model1 = ARIMA(train, order=(1,1,1))
model1_fit = model1.fit()
pred1 = model1_fit.forecast(steps=len(test))

model2 = ARIMA(train, order=(2,1,2))
model2_fit = model2.fit()
pred2 = model2_fit.forecast(steps=len(test))

mse1 = mean_squared_error(test, pred1)
mse2 = mean_squared_error(test, pred2)

print("MSE for ARIMA(1,1,1):", mse1)
print("MSE for ARIMA(2,1,2):", mse2)

if mse1 < mse2:
    print("ARIMA(1,1,1) is better")
else:
    print("ARIMA(2,1,2) is better")
