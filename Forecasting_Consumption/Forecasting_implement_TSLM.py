import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1) 데이터 불러오기
df = pd.read_csv("uschange.csv")

# 2) Index -> 분기 날짜(ds)로 변환
year = df['Index'].astype(float).astype(int)
frac = df['Index'] - year
quarter = frac.map({0.0: 1, 0.25: 2, 0.50: 3, 0.75: 4}).astype(int)

# FutureWarning 피하는 버전
idx = pd.PeriodIndex.from_fields(year=year, quarter=quarter, freq='Q')
df['ds'] = idx.to_timestamp(how='end')  # 분기 말일로 변환

# 3) Prophet용 데이터프레임 (y + regressors)
df_reg = df[['ds', 'Consumption', 'Income', 'Production',
             'Unemployment', 'Savings']].copy()
df_reg = df_reg.rename(columns={'Consumption': 'y'})

# 4) Prophet 모델 정의 + 회귀자 추가
m_reg = Prophet()
m_reg.add_regressor('Income')
m_reg.add_regressor('Production')
m_reg.add_regressor('Unemployment')
m_reg.add_regressor('Savings')

# 5) 모델 학습
m_reg.fit(df_reg)

# 6) 같은 기간에 대해 in-sample 예측
forecast_reg = m_reg.predict(df_reg[['ds', 'Income', 'Production',
                                     'Unemployment', 'Savings']])

df_eval_reg = df_reg.copy()
df_eval_reg['yhat'] = forecast_reg['yhat']

# 7) R² 계산
y = df_eval_reg['y']
yhat = df_eval_reg['yhat']
y_mean = y.mean()

SS_res = ((y - yhat) ** 2).sum()
SS_tot = ((y - y_mean) ** 2).sum()
R2_reg = 1 - SS_res / SS_tot
print("R² (Prophet + regressors):", R2_reg)

# 8) (1) 시계열 플롯: 실제 vs 예측
plt.figure(figsize=(12, 5))
plt.plot(df_eval_reg['ds'], y, label='Actual (Consumption)')
plt.plot(df_eval_reg['ds'], yhat, label='Fitted (Prophet + regressors)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Percent change in Consumption')
plt.title('US Consumption: Actual vs Fitted (Prophet + regressors)')
plt.legend()
plt.tight_layout()
plt.show()

# 9) (2) 산점도 플롯: 실제값 vs 예측값
plt.figure(figsize=(6, 6))
plt.scatter(y, yhat, alpha=0.7)
min_val = min(y.min(), yhat.min())
max_val = max(y.max(), yhat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # y = x 기준선
plt.xlabel('Actual Consumption')
plt.ylabel('Fitted Consumption')
plt.title('Actual vs Fitted (Prophet + regressors)')
plt.tight_layout()
plt.show()
