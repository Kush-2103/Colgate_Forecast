
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

import pandas as pd

file_path = '/content/beauty-personal-care_personal-care_oral-care_india_USD_en.xlsx'
sales_data = pd.read_excel(file_path, sheet_name='Retail')
population_data = pd.read_excel(file_path, sheet_name='Key Market Indicators')

sales_data.head()

population_data.head()

df = pd.DataFrame(population_data)
np.random.seed(42)
df['Sales'] = (df['Population, 0-14 Years in million #'] * 0.1 +
               df['Population, 15-24 Years in million #'] * 0.2 +
               df['Population, 25-34 Years in million #'] * 0.3 +
               df['Population, 35-44 Years in million #'] * 0.25 +
               df['Population, 45-54 Years in million #'] * 0.1 +
               df['Population, 55+ Years in million #'] * 0.05) + np.random.normal(0, 5, len(df))

train = df[df['Year'] < 2002]
test = df[df['Year'] == 2002]

df

# Prepare the feature matrix (X) and target vector (y)
X_train = train[['Population, 0-14 Years', 'Population, 15-24 Years',
                 'Population, 25-34 Years', 'Population, 35-44 Years',
                 'Population, 45-54 Years', 'Population, 55+ Years']]
y_train = train['Sales']
X_test = test[['Population, 0-14 Years', 'Population, 15-24 Years',
               'Population, 25-34 Years', 'Population, 35-44 Years',
               'Population, 45-54 Years', 'Population, 55+ Years']]
y_test = test['Sales']

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_forecast = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_forecast = rf_model.predict(X_test)

# XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_forecast = xgb_model.predict(X_test)

arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=len(y_test))

# Evaluation using error metrics
def evaluate_forecasts(test, forecast, model_name):
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test, forecast)
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

# Evaluate Linear Regression
lr_eval = evaluate_forecasts(y_test, lr_forecast, 'Linear Regression')

# Evaluate Random Forest
rf_eval = evaluate_forecasts(y_test, rf_forecast, 'Random Forest')

# Evaluate XGBoost
xgb_eval = evaluate_forecasts(y_test, xgb_forecast, 'XGBoost')

print(lr_eval)
print(rf_eval)
print(xgb_eval)

