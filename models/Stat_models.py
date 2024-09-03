import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
# from statsmodels.graphics.tsaplots import plot_predict
from prophet import Prophet

class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """

    def __init__(self, configs):
        super(Arima, self).__init__()
        self.pred_len = configs.pred_len

    # def forward(self, x):
    #     result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
    #     intermediate_results = []
    #     for bt,seqs in tqdm(enumerate(x)):
    #         for i in range(seqs.shape[-1]):
    #             seq = seqs[:,i]
    #             timeseries = pd.Series(seq)
    #
    #             # 定义模型参数
    #             order = (1, 1, 1)  # (p, d, q) 其中 p 是自回归项，d 是差分次数，q 是移动平均项
    #             end_of_data = len(timeseries)
    #             # 创建 ARIMA 模型
    #
    #             model = ARIMA(timeseries, order=order,enforce_stationarity=False)
    #             model_fit = model.fit()
    #             prediction = model_fit.predict(start=end_of_data, end=end_of_data + self.pred_len - 1)
    #             result[bt,:,i] = prediction.values
    #
    #     return result # [B, L, D]

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len])
        intermediate_results = []
        for bt,seqs in tqdm(enumerate(x)):

            seq = seqs[:,-1]
            timeseries = pd.Series(seq)

            # 定义模型参数
            order = (1, 1, 1)  # (p, d, q) 其中 p 是自回归项，d 是差分次数，q 是移动平均项
            end_of_data = len(timeseries)
            # 创建 ARIMA 模型

            model = ARIMA(timeseries, order=order,enforce_stationarity=False)
            model_fit = model.fit()
            prediction = model_fit.predict(start=end_of_data, end=end_of_data + self.pred_len - 1)
            result[bt,:] = prediction.values
        result = result.reshape(result.shape[0],result.shape[1],1)

        return result # [B, L, D]

class Prophet(nn.Module):
    def __init__(self, configs):
        super(Prophet, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x, x_mark):
        print(x_mark.shape)
        print(x_mark)
        result = np.zeros([x.shape[0], self.pred_len])
        for bt, seqs in tqdm(enumerate(x)):
            seq = seqs[:, -1]
            timeseries = pd.Series(seq)
            # 创建和拟合模型
            model = Prophet()
            model.fit(timeseries)
            # 创建未来数据的日期框架
            future = model.make_future_dataframe(periods=self.pred_len)

            # 预测未来的数据
            forecast = model.predict(future)
            prediction = forecast['yhat']
            result[bt, :] = prediction.values
        result = result.reshape(result.shape[0], result.shape[1], 1)

        return result  # [B, L, D]

class ETS(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """

    def __init__(self, configs):
        super(ETS, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        result = np.zeros([x.shape[0], self.pred_len])
        intermediate_results = []
        for bt,seqs in tqdm(enumerate(x)):

            seq = seqs[:,-1]
            timeseries = pd.Series(seq)

            # 定义模型参数
            model = ExponentialSmoothing(timeseries, trend="add", seasonal="add", seasonal_periods=10)
            model_fit = model.fit()

            # 进行预测
            prediction = model_fit.forecast(steps=self.pred_len)
            result[bt,:] = prediction.values
        result = result.reshape(result.shape[0],result.shape[1],1)

        return result # [B, L, D]
