# AR모형 기본 코딩 참고자료 : https://direction-f.tistory.com/66
# 자기회귀 주가예측 참고자료 : https://dacon.io/codeshare/2570
# 비계절성, 계절성 Arima 참고자료 : https://otexts.com/fppkr/seasonal-arima.html

# library
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats.distributions import chi2
import pandas as pd

df = pd.read_excel('export.xlsx')
df.head()
df.info()
df.columns

# AR(1)
model_ar_1 = ARIMA(df['한국 수출금액'], order = (1,0,0))
result_ar_1 = model_ar_1.fit()
result_ar_1.summary()