#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:52:37 2018

@author: chenxi
"""

import h5py
import pickle
import numpy as np
from sklearn.externals import joblib


def save_samples_to_h5(sample_x, sample_y, data_path, station, m, ft, sample_type='train'):
    f_y = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_y_{0}.h5'.format(sample_type), 'w')
    f_y.create_dataset('y_{0}'.format(sample_type), data=sample_y)
    f_x_local = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_x_local_{0}.h5'.format(sample_type), 'w')
    for i,x in enumerate(sample_x):
        f_x_local.create_dataset('x_local_{0}'.format(i), data=x)
    f_x_local.close()
    f_y.close()
    return 1


def read_samples_from_h5(data_path, station,m,ft,sample_type='train', IsGlobal=True):
    f_x_local = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_x_local_{0}.h5'.format(sample_type), 'r')
    f_y = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_y_{0}.h5'.format(sample_type), 'r')
    x_local = []
    for i in range(len(f_x_local.keys())):
        key = 'x_local_{0}'.format(i)
        x_local.append(f_x_local[key][:])
    y_sample = f_y['y_{0}'.format(sample_type)][:]
    if IsGlobal:
        x_global = []
        f_x_global = h5py.File(data_path+str(station)+'_'+str(m)+'_'+ft+'_x_global_{0}.h5'.format(sample_type), 'r')
        for j in range(len(f_x_global.keys())):
            key = 'x_global_{0}'.format(j)
            x_global.append(f_x_global[key][:])
        f_x_global.close()
    f_x_local.close()
    f_y.close()
    if IsGlobal:
        return [x_local, x_global], y_sample
    else:
        return x_local, y_sample


def read_scaler(ft,num=3,scaler_path='data/scaler/'):
    num_dict = {1:'01',2:'02', 3:'03'}
    scaler_path = scaler_path+'scaler'+num_dict[num]+'_'+ft
    min_max_scaler = joblib.load(scaler_path)
    return min_max_scaler


def save_pkl(file_name, file):
    saved_file = open(file_name, 'wb')
    pickle.dump(file, saved_file, pickle.HIGHEST_PROTOCOL)
    saved_file.close()


def load_pkl(file_name):
    file = open(file_name, 'rb')
    a = pickle.load(file)
    file.close()
    return a


def data_transform(x_train):
    x_encoder_nume = x_train[0]
    x_encoder_category = [x_train[i] for i in [1, 2, 3, 4]]
    x_encoder_category = np.concatenate(x_encoder_category, axis=-1).astype(np.int)
    y_init = x_train[5]
    x_decoder_nume = x_train[6]
    x_decoder_category = [x_train[i] for i in [7,8,9]]
    x_decoder_category = np.concatenate(x_decoder_category, axis=-1).astype(np.int)
    return x_encoder_nume, x_encoder_category, x_decoder_nume, x_decoder_category, y_init
