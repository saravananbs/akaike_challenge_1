import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

data_path = 'data/data.csv'
data = pd.read_csv(data_path, encoding="ISO-8859-1")

data.fillna({"CustomerID": 'No CustomerID'}, inplace=True)

data["Revenue"] = data["Quantity"] * data["UnitPrice"]

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

daily_sales = data.groupby(data["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
daily_sales.columns = ['InvoiceDate', 'Revenue']

plt.figure(figsize=(12, 6))
plt.plot(daily_sales['InvoiceDate'], daily_sales['Revenue'])
plt.title('sales trend over time')
plt.xlabel('date')
plt.ylabel('revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def create_lagged(df, lag=1):
    lagged = df.copy()
    for i in range(1, lag+1):
        lagged[f'lag_{i}'] = lagged['Revenue'].shift(i)
    return lagged

lag = 5
sales_lag = create_lagged(daily_sales, lag)
sales_lag = sales_lag.dropna()

X = sales_lag.drop(columns=['InvoiceDate', 'Revenue'])
y = sales_lag['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, pred))
print(f"RMSE: {error:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='actual sales', color='red')
plt.plot(y_test.index, pred, label='predicted sales', color='green')
plt.title('sales forecasting using XGBoost')
plt.xlabel('index')
plt.ylabel('revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

model.fit(X, y)

future_preds = []
last_known = sales_lag.iloc[-1][[f'lag_{i}' for i in range(1, lag+1)]].values.tolist()
for i in range(7):
    input_features = np.array(last_known[-lag:]).reshape(1, -1)
    pred = model.predict(input_features)[0]
    future_preds.append(pred)
    last_known.append(pred)

last = sales_lag['InvoiceDate'].iloc[-1]
future = pd.date_range(last + pd.Timedelta(days=1), periods=7)

plt.figure(figsize=(14, 7))
plt.plot(sales_lag['InvoiceDate'], sales_lag['Revenue'], label='Actual Sales', color='red')
plt.plot(future, future_preds, label='Forecasted Sales (Next 7 Days)', color='blue', marker='o')
plt.title('actual vs forecasted daily sales')
plt.xlabel('date')
plt.ylabel('revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()