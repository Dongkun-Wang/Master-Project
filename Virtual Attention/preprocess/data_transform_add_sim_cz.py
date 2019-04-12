# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:45:16 2018

@author: xchen
"""
import datetime as dt
import pandas as pd
import numpy as np
from cfg import *
import pymysql
from dateutil.parser import parse
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.externals import joblib
class add_statistical_features(object):
    '''
    添加各种统计值的函数类
    '''
    def add_ma(self, data_df, ft, ma_window_list):
        '''
        计算均值，天数等于ma_window
        '''
        for ma_window in ma_window_list:
            ma_window = int(ma_window)
            data_df[ft+'_ma_'+str(ma_window)] = data_df[ft].rolling(window=ma_window).mean()
        return data_df

    def add_gradient(self,data_df, ft):
        '''
        计算变化率
        '''     
        temp_list = list(data_df[ft])
        temp_list = temp_list[:1]+temp_list[:-1]
        data_df[ft+'_gradient'] = temp_list
        data_df[ft+'_gradient'] = data_df[ft] - data_df[ft+'_gradient']
        return data_df
        
    def add_ema(self,data_df,ft,ema_window_list):
        '''
        计算指数滑动平均值，天数等于ema_window
        '''
        for ema_window in ema_window_list:
            ema_window = int(ema_window)
            data_df[ft+'_ema_'+str(ema_window)] = data_df[ft].ewm(span=ema_window, min_periods=ema_window).mean()
        return data_df
        
def revalue(x, labels):
    if isinstance(x, float) and np.isnan(x):
        return x
    else:
        return labels.index(x)
        
def label_encoder(series):
    labels = list(series.unique())
    labels = [l for l in labels if not (isinstance(l, float) and np.isnan(l))]
    return series.map(lambda x: revalue(x, labels))
    
def angle2dir(angle):

    # convert the angle to the direction
    # angle: [0, 359], direction: [1, 8]
    ranges = {
        (0, 45): 0,
        (45, 90): 1,
        (90, 135): 2,
        (135, 180): 3,
        (180, 225): 4,
        (225, 270): 5,
        (270, 315): 6,
        (315, 360): 7
    }
    if np.isnan(angle) or angle == 999999:
        return angle
    else:
        angle = int(angle) % 360
        for k in ranges.keys():
            if k[0] <= angle < k[1]:
                return ranges[k]


class create_aq_meo(add_statistical_features):

    def __call__(self,station_id):
        '''
        return station_aq_meo        
        '''
        return self.get_station_aq_meo(station_id)
    
    def get_time_df(self, start_time, end_time):
        '''
        get integrated time_df from start_time to end_time
        '''
        delta_hour = dt.timedelta(hours=1)
        time_df = [start_time]
        while time_df[-1] < end_time:
            time_df.append((parse(time_df[-1])+delta_hour).strftime('%Y-%m-%d %H:%M:%S'))
        return pd.DataFrame(time_df,columns=['time'])
    
    def modify_time(self, time):
        time = time.strftime('%Y-%m-%d %H:%M:%S')
        minute = int(time[-5:-3])
        time_temp = parse(time[:-6])
        if minute > 30:
            time_temp += dt.timedelta(hours=1)
        time = time_temp.strftime('%Y-%m-%d %H:%M:%S')
        return time
        
    def add_time_week_info(self, aq_meo_df):
        '''
        add hour, day informations into dataframe
        '''
        aq_meo_df['day_time'] = aq_meo_df['time'].map(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
        aq_meo_df['week_day'] = aq_meo_df['time'].map(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').weekday())
        return aq_meo_df
            
    def get_chem(self, station_id):
        '''
        get station airquality
        '''
        conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
        chem = pd.read_sql('select * from observation_chem_site where file_date < %(date)s', con=conn,
                            params={'date': ph.end_date})
        conn.close()
        chem_site = chem[chem['site_id']==station_id]
        chem_site_list = []
        for ft in ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']:
            chem_site_ft = chem_site[chem_site['chem_type']==ft]
            chem_site_ft = chem_site_ft[['site_id','observed_time','chem_value']]
            chem_site_ft.columns = ['station_id','time',ft]
            chem_site_ft = chem_site_ft.reset_index(drop=True)
            chem_site_list.append(chem_site_ft)
        chem_site = pd.concat(chem_site_list, axis=1)
        chem_site = chem_site.groupby(level=0,axis=1).first()
        chem_site = chem_site.drop_duplicates(subset='time')
        chem_site['time'] = chem_site['time'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        return chem_site

    def get_met(self, city_id):
        '''
        get station met
        '''
        conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
        met = pd.read_sql('select * from observation_met_city where file_date < %(date)s', con=conn,
                           params={'date': ph.end_date})
        conn.close()
        met_site = met[met['city_id'] == city_id]
        met_site['observed_time'] = met_site['observed_time'].map(lambda x: self.modify_time(x))
        met_site_list = []
        for ft in ['temperature', 'rh','wd','wp','weather']:
            met_site_ft = met_site[met_site['met_type']==ft]
            met_site_ft = met_site_ft[['observed_time','met_value']]
            met_site_ft.columns = ['time',ft]
            met_site_ft = met_site_ft.reset_index(drop=True)
            met_site_list.append(met_site_ft)
        met_site = pd.concat(met_site_list, axis=1)
        met_site = met_site.groupby(level=0,axis=1).first()
        met_site = met_site.drop_duplicates(subset='time')
        met_site['rh'] = met_site['rh'].map(lambda x: float(x)/100)
        met_site['temperature'] = met_site['temperature'].map(lambda x: float(x))
        met_site['wd'] = met_site['wd'].map(lambda x: wind_encoder(x))
        met_site['weather'] = met_site['weather'].map(lambda x: weather_encoder(x))
        met_site['wp'] = met_site['wp'].map(lambda x: wind_speed_decoder(x))
        met_site.columns = ['humidity', 'temperature', 'time', 'wind_direction', 'weather', 'wind_speed']
        return met_site

    def get_simulation(self, station_id):
        model_type = 'Mix'
        conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
        fore_chem = pd.read_sql('select * from forecast_chem_site where file_date < %(date)s', con=conn,
                          params={'date': ph.end_date})
        conn.close()
        fore_chem_site = fore_chem[fore_chem['site_id']==station_id]
        fore_chem_site = fore_chem_site[fore_chem_site['model']==model_type]
        fore_chem_site_list = []
        for ft in ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']:
            fore_chem_site_ft = fore_chem_site[fore_chem_site['chem_type']==ft]
            fore_chem_site_ft = fore_chem_site_ft[['site_id','forecast_time','chem_value']]
            fore_chem_site_ft.columns = ['station_id','time',ft+'_SIM']
            fore_chem_site_ft = fore_chem_site_ft.reset_index(drop=True)
            fore_chem_site_list.append(fore_chem_site_ft)
        fore_chem_site = pd.concat(fore_chem_site_list, axis=1)
        fore_chem_site = fore_chem_site.groupby(level=0,axis=1).first()
        fore_chem_site = fore_chem_site.drop_duplicates(subset='time')
        fore_chem_site['time'] = fore_chem_site['time'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        
        return fore_chem_site
            
    def get_chuzhou_aq_meo_with_sim(self, station_id):
        city_id = 341100
        cz_chem = self.get_chem(station_id)
        cz_met = self.get_met(city_id)
        cz_aq_meo = pd.merge(cz_chem,cz_met, how='outer', on=['time'])
        cz_sim = self.get_simulation(station_id)
        cz_aq_meo = self.add_time_week_info(cz_aq_meo)
        cz_aq_meo = pd.merge(cz_aq_meo, cz_sim, on=['time','station_id'], how='outer')
        for ft in ['PM2.5','PM10','NO2','CO','O3','SO2']:
            cz_aq_meo = self.add_ma(cz_aq_meo, ft , ma_window_list=[6,12,24])
            cz_aq_meo = self.add_ema(cz_aq_meo, ft , ema_window_list=[6,12,24])
            cz_aq_meo = self.add_gradient(cz_aq_meo, ft)
        return cz_aq_meo


def read_scaler(ft,num=1):
        # scaler_path = 'data/scaler/'
        real_path = os.path.dirname(os.path.realpath(__file__))
        print('real_path:', real_path)
        scaler_path = real_path + '/../data/scaler/'
#        ft_dict = {'pm2.5':'PM2.5',
#                   'pm10':'PM10',
#                   'o3':'O3',
#                   'no2':'NO2',
#                   'co':'CO',
#                   'so2':'SO2'}   
        num_dict = {1:'01',2:'02',3:'03'}
        scaler_path = scaler_path+'scaler'+num_dict[num]+'_'+ft
        min_max_scaler = joblib.load(scaler_path)
        return min_max_scaler

        
class data_preprocess(create_aq_meo):
    def __init__(self,ft, 
                 check_columns,
                 num_columns = ['temperature', 'humidity', 'wind_speed'],
                 end_date='2015-04-01'):
        self.ft = ft  
        self.check_columns = check_columns
        self.num_columns = num_columns
        self.end_date = end_date
        self.processes_num = 8                   # pollutant type
#        self.cz_stations = ['监测站','人大宾馆','老年大学']
        self.cz_stations = [2298,2299,2300]

        super(data_preprocess, self).__init__()
        
    def __call__(self, station_id=None, return_all=False, data_type='train'):
        '''
        read all stations info, scaled with all stations numerical info
        '''
#        bj_aq_meo = self.multiprocess(self.get_station_aq_meo,\
#                                        self.aq_stations, self.processes_num)
        cz_aq_meo = self.multiprocess(self.get_chuzhou_aq_meo_with_sim,\
                                           self.cz_stations, self.processes_num)
        self.aq_meo = cz_aq_meo
        print('DONE')       
        # train or validate
        if data_type == 'train' or data_type == 'training':
            self.aq_meo = self.aq_meo[self.aq_meo['time']<self.end_date]
        elif data_type == 'test' or data_type == 'testing':
            self.aq_meo = self.aq_meo[self.aq_meo['time']>=self.end_date]
#            self.aq_meo = self.aq_meo[self.aq_meo['time']<'2018-10-01']
        else:
            raise ValueError('data_type must be train or test')
        print('DONE!')
        self.aq_meo = self.aq_meo.set_index('time',drop=False)
        numeric_data, statistical_data, categorical_data, simulation_data = self.preprocessing(self.aq_meo)
        if data_type == 'train' or data_type == 'training':
            scaler1, scaler2, scaler3 = self.get_scalers(numeric_data, statistical_data, categorical_data)
        elif data_type == 'test' or data_type == 'testing':
            scaler1 = read_scaler(self.ft,1)
            scaler2 = read_scaler(self.ft,2)
            scaler3 = read_scaler(self.ft,3)
        else:
            pass
        # return station_id or all stations   
        if return_all:
            pass
        elif station_id:       
            self.aq_meo = self.aq_meo[self.aq_meo['station_id']==station_id]
            numeric_data = numeric_data[numeric_data['station_id']==station_id]
            statistical_data = statistical_data[statistical_data['station_id']==station_id]
            categorical_data = categorical_data[categorical_data['station_id']==station_id]   
            simulation_data = simulation_data[simulation_data['station_id']==station_id]   
        else:
            raise ValueError('station_id is None')
        return numeric_data, statistical_data, categorical_data, simulation_data, scaler1, scaler2, scaler3
                        
    def get_scalers(self, numeric_data, statistical_data, categorical_data):
        '''
        create scalers
        scaler1: numeric data
        scaler2: statistical data
        scaler3: ft
        '''
        real_path = os.path.dirname(os.path.realpath(__file__))
        print('real_path:', real_path)
        scaler_path = real_path + '/../data/scaler/'
        numeric_data_for_scaler = self.fill_num_nans(numeric_data[self.num_columns])
        statistical_columns = [col for col in statistical_data.columns if self.ft in col]
        statistical_data_for_scaler = self.fill_num_nans(statistical_data[statistical_columns])        
        scaler1 = self.scale_xy(numeric_data_for_scaler)
        scaler2 = self.scale_xy(statistical_data_for_scaler)
        scaler3 = self.scale_xy(numeric_data_for_scaler[[self.ft]])
        joblib.dump(scaler1, scaler_path + 'scaler_01'+self.ft)
        joblib.dump(scaler2, scaler_path + 'scaler_02' + self.ft)
        joblib.dump(scaler3, scaler_path + 'scaler_03' + self.ft)
        return scaler1, scaler2, scaler3        
        
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
        
    def scale_xy(self, x):
        '''
        create scalers
        '''
        scaler1 = MinMaxScaler()
        scaler1.fit_transform(x)
        return scaler1
                   
        
    def fill_num_nans(self, num_sample, axis=0):
        return num_sample.interpolate(method='linear', axis=axis).ffill().bfill()                

    def fill_cat_nans(self, cat_sample):
        return cat_sample.ffill().bfill()
 
    def preprocessing(self, aq_meo):
        # 过滤不需要的列, fill nans not in self.numeric_columns
        aq_meo = self.add_time_week_info(aq_meo)
        aq_meo = aq_meo.drop_duplicates(['station_id', 'time'])
        # deal with missing values and 999999
        aq_meo = aq_meo.replace(999999, np.nan)
        numeric_data = aq_meo[['station_id','PM2.5','PM10','O3','NO2','CO','SO2']+\
                             ['temperature', 'humidity', 'wind_speed']]
        simulation_data = aq_meo[['station_id', 'PM2.5_SIM', 'PM10_SIM', 'SO2_SIM', 'NO2_SIM', 'CO_SIM', 'O3_SIM']]
        for column in numeric_data.columns:
            if column not in self.check_columns:
                numeric_data[column] = self.fill_num_nans(numeric_data[column])                                                 
        statistical_data = aq_meo[['station_id','PM2.5_ma_6','PM2.5_ma_12','PM2.5_ma_24']+\
                                   ['PM2.5_ema_6','PM2.5_ema_12','PM2.5_ema_24', 'PM2.5_gradient']+\
                                   ['PM10_ma_6','PM10_ma_12','PM10_ma_24']+\
                                   ['PM10_ema_6','PM10_ema_12','PM10_ema_24', 'PM10_gradient']+\
                                   ['NO2_ma_6','NO2_ma_12','NO2_ma_24']+\
                                   ['NO2_ema_6','NO2_ema_12','NO2_ema_24', 'NO2_gradient']+\
                                   ['CO_ma_6','CO_ma_12','CO_ma_24']+\
                                   ['CO_ema_6','CO_ema_12','CO_ema_24', 'CO_gradient']+\
                                   ['O3_ma_6','O3_ma_12','O3_ma_24']+\
                                   ['O3_ema_6','O3_ema_12','O3_ema_24', 'O3_gradient']+\
                                   ['SO2_ma_6','SO2_ma_12','SO2_ma_24']+\
                                   ['SO2_ema_6','SO2_ema_12','SO2_ema_24', 'SO2_gradient']]     
        statistical_data = self.fill_num_nans(statistical_data)
        categorical_data = aq_meo[['station_id','wind_direction','weather','day_time','week_day']]
        categorical_data = self.fill_cat_nans(categorical_data)
        simulation_data = self.fill_num_nans(simulation_data)
        print(aq_meo[['time']][-1:])

        return numeric_data, statistical_data, categorical_data, simulation_data


if __name__ == '__main__':
    ft_list = ['PM2.5','PM10','O3','NO2', 'CO', 'SO2']
    ft = 'PM10'
    end_date = '2018-09-31'
    station_list = [2298]
    for ft in ft_list[1:2]:
        data_preprocessing = data_preprocess(ft,[ft],[ft,'temperature', 'humidity', 'wind_speed'], end_date)
        for station_id in station_list:
            numeric_data, statistical_data, categorical_data, simulation_data, scaler1, scaler2, scaler3 = \
                    data_preprocessing(station_id=station_id,
                                       return_all=False, data_type='train')

