# -*- coding: utf-8 -*-
from preprocess.data_transform_add_sim_cz import data_preprocess
from cfg import *
import pandas as pd
import numpy as np
from preprocess.tools import save_samples_to_h5


class Functions(object):

    def split_train_test_samples(self, samples_list, y_samples, train_percent=0.9):
        # split train and validate samples

        if isinstance(samples_list, tuple):
            IsGlobal = True
            target_staiton_samples, rest_stations_samples = samples_list
        else:
            IsGlobal =False
            target_staiton_samples = samples_list
        if IsGlobal:
            lens = int(y_samples.shape[0]*train_percent)
            x_train = []
            x_test =[]
            for i in range(len(target_staiton_samples)):
                x_train.append(target_staiton_samples[i][:lens])
                x_test.append(target_staiton_samples[i][lens:])
            x_train = [x_train] + [rest_stations_samples[:lens]]
            x_test = [x_test] + [rest_stations_samples[lens:]]
            
            return x_train, y_samples[:lens], x_test, y_samples[lens:]
        else:
            lens = int(y_samples.shape[0]*train_percent)
            x_train = []
            x_test = []
            for i in range(len(target_staiton_samples)):
                x_train.append(target_staiton_samples[i][:lens])
                x_test.append(target_staiton_samples[i][lens:])
            return x_train, y_samples[:lens], x_test, y_samples[lens:]              
        
    def big_nan_filter(self, num_cat_data, check_columns, percent = 0.5):
        '''
        count nans in count_columns, if max percent > percent, drop it
        '''
        if len(num_cat_data.columns)==1:
            check_columns = [self.ft]
        filted_data = num_cat_data[check_columns]
        max_nan_count = np.max(filted_data.isnull().sum(axis=0))
        if max_nan_count/len(filted_data) > percent:
            return pd.DataFrame([])
        else:
            return num_cat_data
            
    def multiprocess(self, func, variable_list, processes_num):
        '''
        multiprocessing module
        '''
        from multiprocessing import Pool
        pool = Pool(processes=processes_num)
        pool_out = pool.map(func, variable_list)
        pool.close()
        pool.join()  
        return pd.concat(pool_out)
    
    def fill_num_nans(self, num_sample, axis=0):
        return num_sample.interpolate(method='linear', axis=axis).ffill().bfill()                

    def fill_cat_nans(self, cat_sample):
        return cat_sample.ffill().bfill()
        
    def add_noise_to_num_forecast(self, num_forecast_array, noise_scale=0.01):
        '''
        num_forcast_columns: ft, pressure, temperature, humidity, wind_speed
        '''
        num_forecast_array = num_forecast_array[:,:,1:] + \
                            np.random.normal(0,scale=noise_scale,size=num_forecast_array[:,:,1:].shape)
                            
        return num_forecast_array


class CreateDataset(Functions):
    def __init__(self, ft,  statistical_list, noise_scale=0.001, end_date='2018-10-01'):
        self.train_percent = None
        self.ft = ft
        self.m = hp.time_step                                    # input time length
        self.n = hp.output_time_step                        # output time length
        self.noise_scale = noise_scale
        self.statistical_list = statistical_list
        if self.ft == 'PM2.5':
            self.numeric_columns = [self.ft,'NO2','SO2'] + ['temperature','humidity','wind_speed']
            self.check_columns = [self.ft,'NO2','SO2'] + ['temperature','humidity']
        else:
            self.numeric_columns = [self.ft] + ['temperature','humidity','wind_speed']
            self.check_columns = [self.ft] + ['temperature','humidity']
        self.data_preprocessing = data_preprocess(self.ft, self.check_columns,
                                                 self.numeric_columns, end_date)
    
    def __call__(self,station_id, inputs_type, data_type):
        '''
        local x_train: data list, including [numeric_data, categorical_data_list, y_initial, 
                                             numeric_forecast, categorical_forecast]
        global x_train: data list including [local x_train, rest_stations_numeric_data]
        '''
        self.station_id = station_id
        if inputs_type == 'local':
            numeric_data, statistical_data, categorical_data, simulation_data,\
                    scaler1, scaler2, scaler3 = self.data_preprocessing(station_id=self.station_id,data_type=data_type)
            numeric_data_station, statistical_data_station, categorical_data_station, simulation_data_station = \
                            self.get_station_data(numeric_data, statistical_data, categorical_data, simulation_data, self.station_id)
            samples_list,y_samples = \
                            self.loop_create_samples(numeric_data_station, statistical_data, categorical_data_station,
                                                     simulation_data_station, scaler1, scaler2, scaler3)
            if data_type == 'test':
                return samples_list, y_samples, scaler3
            else:
                x_train, y_train, x_test, y_test = self.split_train_test_samples(samples_list,y_samples)            
                return x_train, y_train, x_test, y_test, scaler3
#            return samples_list,y_samples
            
        elif inputs_type == 'global': 
            numeric_data, statistical_data, categorical_data, \
                    scaler1, scaler2, scaler3 = self.data_preprocessing(return_all=True)
            target_station_samples, rest_stations_samples, y_samples = \
                            self.get_global_samples(numeric_data, statistical_data, categorical_data, 
                                                    scaler1, scaler2, scaler3)
            x_train, y_train, x_test, y_test = self.split_train_test_samples((target_station_samples,
                                                                              rest_stations_samples),
                                                                              y_samples)                        
            return x_train, y_train, x_test, y_test, scaler3
        else:
            raise TypeError('input_type must be "local" or "global"!')

    def get_global_samples(self,numeric_data, statistical_data, categorical_data, scaler1, scaler2, scaler3):
        '''

        get all stations info
        split datas into samples, filter big nans
        return target_staiton_samples,rest_stations_samples,y_samples
        shape: target_staiton_samples([numerical_data, categorical_data1,categorical_data2, ..., categorical_dataN])
               rest_stations_samples(samples, stations, in_timestep, 1)
               y_samples(samples,out_timestep,1)
        '''
        stations, _ = self.data_preprocessing.get_aq_stations()
        stations = stations
        stations_data_dict = {}
        for station in stations:
            stations_data_dict[station] = self.get_station_data(numeric_data,statistical_data,categorical_data,station)
        
        target_num_data, target_sta_data, target_cat_data = stations_data_dict[self.station_id]
        # rest_num_data: only ft of stations
        rest_num_data, _, _ = self.get_rest_stations_data(stations_data_dict, stations, self.station_id)
        target_staiton_samples, rest_stations_samples, y_samples= \
                        self.loop_create_samples([target_num_data,rest_num_data], target_sta_data, target_cat_data,\
                                                 scaler1, scaler2, scaler3)
        return target_staiton_samples,rest_stations_samples,y_samples
     
    def loop_create_samples(self, numeric_data, statistical_data, categorical_data, simulation_data, scaler1, scaler2, scaler3):
        if isinstance(numeric_data, list):
            IsGlobal = True
            target_numeric_data, rest_numeric_data = numeric_data
            target_statistical_data = statistical_data
            target_categorical_data = categorical_data
        else:
            IsGlobal =False
            target_numeric_data = numeric_data
            target_statistical_data = statistical_data            
            target_categorical_data = categorical_data    
        target_numeric_data = target_numeric_data[['PM2.5','PM10','O3','NO2','CO','SO2']+\
                                         ['temperature', 'humidity', 'wind_speed']]
        target_categorical_data = target_categorical_data[['wind_direction','weather','day_time','week_day']]                                 
        simulation_data.columns = ['PM2.5','PM10','SO2','NO2','CO','O3']
        sample_list = [[],[],[],[],[],[],[]]
        scaler_dict = {
                       0:scaler1,       #num_sample
                       1:scaler2,       #sta_sample
                       3:scaler3,       #y_initial
                       4:scaler1,       #num_forcast
                       6:scaler3,       #y
                       }
        if IsGlobal:
            rest_sample_list = []       #num_sample_list
            rest_numeric_data = self.fill_num_nans(rest_numeric_data)
#            print('rest_numeric_data shape:', rest_numeric_data.shape)
        sta_col_name = [col for col in target_statistical_data.columns if self.ft in col]
        target_statistical_data = target_statistical_data[sta_col_name]
        target_numeric_data = target_numeric_data[self.numeric_columns]
#        print('target_numeric_data shape:', target_numeric_data.shape)
        i = self.m
        while i< len(target_numeric_data)-self.n:
            num_sample = target_numeric_data.iloc[i-self.m:i]
            sta_sample = target_statistical_data.iloc[i-self.m:i]
            cat_sample = target_categorical_data.iloc[i-self.m:i]
            y = target_numeric_data[[self.ft]].iloc[i:i+self.n]
            num_forecast = target_numeric_data.iloc[i:i+self.n]
            cat_forecast = target_categorical_data[['wind_direction','day_time','week_day']].iloc[i:i+self.n]
            num_sample = self.big_nan_filter(num_sample, self.check_columns, percent=0.5)
            y = self.big_nan_filter(y, self.check_columns, percent=0.5)
            num_forecast = self.big_nan_filter(num_forecast, self.check_columns, percent=0.5)
            if IsGlobal:
                rest_num_sample = rest_numeric_data.iloc[i-self.m:i]
            i+=1
            if len(num_sample) and len(y) and  len(num_forecast):           #nans is not enough to drop
                target_num_sample = self.fill_num_nans(num_sample)          #fill numerical nans with linear method
                target_sta_sample = self.fill_num_nans(sta_sample)          #fill statistical nans with linear method
                target_cat_sample = self.fill_cat_nans(cat_sample)          #fill categorical nans
                num_forecast = self.fill_num_nans(num_forecast)
                cat_forecast = self.fill_cat_nans(cat_forecast)
                y = self.fill_num_nans(y)
#                y_initial = pd.concat([target_num_sample[[self.ft]].iloc[-1] for _ in range(self.n)],axis=0)
                y_initial = simulation_data[[self.ft]].iloc[i:i+self.n]
                sample_list[0].append(np.array(target_num_sample).reshape(1,self.m,-1))         #append target_num_samples
                sample_list[1].append(np.array(target_sta_sample).reshape(1,self.m,-1))         #append target_sta_samples
                sample_list[2].append(np.array(target_cat_sample).reshape(1,self.m,-1))         #append target_cat_samples
                sample_list[3].append(np.array(y_initial).reshape(1,self.n,1))                  #append y_initial
                sample_list[4].append(np.array(num_forecast).reshape(1,self.n,-1))              #append num_forecast
                sample_list[5].append(np.array(cat_forecast).reshape(1,self.n,-1))              #append cat_forecast
                sample_list[6].append(np.array(y).reshape(1,self.n,1))                          #append y
                if IsGlobal:
                    rest_num_sample = np.array(rest_num_sample.T)
                    rest_sample_list.append(rest_num_sample[np.newaxis,:,:,np.newaxis])
#        print('rest_sample_list length:', len(rest_sample_list))
        for j in range(len(sample_list)):
            print(j)
            sample_list[j] = np.concatenate(sample_list[j],axis=0)
            sample_shape = sample_list[j].shape
            if j not in [2,5]:               
                sample_list[j] = scaler_dict[j].transform(sample_list[j].reshape(-1,sample_shape[-1])).reshape(sample_shape)
        # add noise to num_forecast and drop ft in num_forecast
        sample_list[4] = self.add_noise_to_num_forecast(sample_list[4],noise_scale=self.noise_scale)    
        if IsGlobal:
            rest_sample_list = np.concatenate(rest_sample_list,axis=0)
            rest_sample_shape = rest_sample_list.shape
            rest_sample_list = scaler3.transform(rest_sample_list.reshape(-1,1)).reshape(rest_sample_shape)
        cat_samples_list = []
        cat_forecast_list = []
        # append target_cat_samples
        for dim1 in range(sample_list[2].shape[-1]):
            cat_samples_list.append(sample_list[2][:,:,dim1][:,:,np.newaxis])
        # append cat_forecast_samples
        for dim2 in range(sample_list[5].shape[-1]):
            cat_forecast_list.append(sample_list[5][:,:,dim2][:,:,np.newaxis])
        samples_list = [np.concatenate((sample_list[0],sample_list[1]),axis=-1)] + cat_samples_list + \
                        [sample_list[3],sample_list[4]] + cat_forecast_list
        if IsGlobal:
            return samples_list, rest_sample_list, sample_list[-1]
        else:
            return samples_list, sample_list[-1]

    def get_station_data(self, numeric_data, statistical_data, categorical_data, simulation_data, station_id):
        numeric_data_station = numeric_data[numeric_data['station_id']==station_id]
        del numeric_data_station['station_id']
        statistical_data_station = statistical_data[statistical_data['station_id']==station_id]
        del statistical_data_station['station_id']
        simulation_data_station = simulation_data[simulation_data['station_id']==station_id]
        del simulation_data_station['station_id']
        categorical_data_station = categorical_data[categorical_data['station_id']==station_id]
        return numeric_data_station, statistical_data_station, categorical_data_station,simulation_data_station
        
    def get_rest_stations_data(self, stations_dict, station_list, station_id):
        '''
        stations_dict: dictionary of stations_info
        '''
        station_list.remove(station_id)
        num_list = []
        sta_list = []
        cat_list = []
        for station in station_list:
            num, sta, cat = stations_dict[station]
            num_list.append(num[[self.ft]].rename(columns={self.ft: '{0}_{1}'.format(self.ft, station)}))
            sta_list.append(sta)
            cat_list.append(cat)
        nums = pd.concat(num_list,axis=1)
        stas = pd.concat(sta_list,axis=1)
        cats = pd.concat(cat_list,axis=1)
        return nums, stas, cats


def concate_samples(sample_list):
    samples = []
    for i in range(len(sample_list[0])):
        samples.append(np.concatenate([sample_list[j][i] for j in range(len(sample_list))]))
    return samples


if __name__ == '__main__':
    ft_list = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']
    noise_scale = 0
    end_date = ph.end_date
    data_path = 'data/train/'
    _station = ph.name
    stations = [2298, 2299, 2300]
    for ft in ft_list:
        print('current ft:', ft)
        a = CreateDataset(ft, ['ema', 'gradient'], noise_scale, end_date)
        x_train_list = []
        y_train_list = []
        x_test_list = []
        y_test_list = []
        for station in stations:
            x_train, y_train, x_test, y_test, scaler = a(station, 'local', 'train')
            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_test_list.append(x_test)
            y_test_list.append(y_test)
        x_train = concate_samples(x_train_list)
        y_train = np.concatenate(y_train_list)
        x_test = concate_samples(x_test_list)
        y_test = np.concatenate(y_test_list)
        save_samples_to_h5(x_train, y_train, data_path, _station, hp.time_step, ft, sample_type='train')
        save_samples_to_h5(x_test, y_test, data_path, _station, hp.time_step, ft, sample_type='test')

