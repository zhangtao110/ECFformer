import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm



# 这里我们假设 p=2, d=1, q=2
model = ARIMA(data, order=(2, 1, 2))
model_fit = model.fit()

# 打印模型摘要
print(model_fit.summary())
# 预测未来5个数据点
forecast = model_fit.forecast(steps=5)
print(forecast)
