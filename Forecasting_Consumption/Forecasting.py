import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("uschange.csv")

# Index → 날짜 변환
year = df['Index'].astype(float).astype(int)
frac = df['Index'] - year
quarter = frac.map({0.0: 1, 0.25: 2, 0.50: 3, 0.75: 4}).astype(int)

idx = pd.PeriodIndex(year=year, quarter=quarter, freq='Q')
df['ds'] = idx.to_timestamp(how='end')

# Prophet 형식으로 변환
df_prophet = df[['ds', 'Consumption']].rename(columns={'Consumption': 'y'})

m = Prophet()
m.fit(df_prophet)

forecast = m.predict(df_prophet[['ds']])  # 기존 날짜만 예측
df_eval = df_prophet.copy()
df_eval['yhat'] = forecast['yhat']

y = df_eval['y']
yhat = df_eval['yhat']
y_mean = y.mean()

SS_res = ((y - yhat) ** 2).sum()
SS_tot = ((y - y_mean) ** 2).sum()

R2 = 1 - SS_res / SS_tot
print("R²:", R2)


plt.figure(figsize=(12,6))
plt.plot(df_eval['ds'], y, label='Actual')
plt.plot(df_eval['ds'], yhat, label='Fitted')
plt.legend()
plt.title("Prophet Fitted vs Actual")
plt.show()
