# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def scal(x):
    scaler = MinMaxScaler(feature_range=(0.1, 1))
    scaler.fit_transform(x)
    return scaler


class CreateStockSamples(object):
    def __init__(self, input_time_step, output_time_step):
        self.file_path = 'data/nasdaq100.csv'
        self.m = input_time_step
        self.n = output_time_step

    def create_samples(self, sample_type):
        stock_df = pd.read_csv(self.file_path)
        stock_name = stock_df.columns.tolist()
        stock_name.remove('NDX')
        ndx_index = stock_df[['NDX']]
        stock_index = stock_df[stock_name]
        scaler_x = scal(stock_index)
        scaler_y = scal(ndx_index)
        train_x = []
        train_y = []
        if sample_type == 'train':
            start = self.m
            end = 35100 + self.m
        elif sample_type == 'valid':
            start = 35100 + self.m
            end = 35100+2730 + self.m
        elif sample_type == 'test':
            start = 35100+2730 - self.n
            end = 35100+2730*2 - self.n
        else:
            raise TypeError('sample_type must be train, valid or test!')
        for i in range(start, end):
            x_temp = stock_index.iloc[i-self.m:i]
            y_temp = ndx_index.iloc[i: i+self.n]
            train_x.append(np.array(scaler_x.transform(x_temp)).reshape(1, -1, len(stock_name)))
            train_y.append(np.array(scaler_y.transform(y_temp)).reshape(1, -1))
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        return train_x, train_y, scaler_y




