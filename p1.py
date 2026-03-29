# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

ts = data['value']

# Step 4: Plot original data
plt.plot(ts)
plt.title("Original Time Series")
plt.show()

# Step 5: Apply ARIMA model (p, d, q)
model = ARIMA(ts, order=(1,1,1))   # simplest commonly used
model_fit = model.fit()

# Step 6: Print summary
print(model_fit.summary())

# Step 7: Forecast future values
forecast = model_fit.forecast(steps=5)

print("Forecasted Values:")
print(forecast)

# Step 8: Plot forecast
plt.plot(ts, label='Original')
plt.plot(range(len(ts), len(ts)+5), forecast, color='red', label='Forecast')
plt.legend()
plt.show()
