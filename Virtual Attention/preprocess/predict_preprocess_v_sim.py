# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:26:11 2018
@author: xchen
"""

import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime as dt
from cfg import*
import pymysql


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

def cal_wind_direction(v1,v2):
    '''
    v1: positive north, negative sourth
    v2: positive east, negetive west
    '''
    import math
    v = (v1**2 + v2**2)**0.5
    if v2 > 0:
        angle = math.degrees(math.acos(v1/v))
    else:
        angle = 360 - math.degrees(math.acos(v1/v))
    return angle2dir(angle)        


def fill_missing_values_numeric(df,axis=0):

    # linear interpolation
    return df.interpolate(method='linear', axis=axis).ffill().bfill()


def changeTimeStamp(time):
    # 改变meo_file的时间戳
    time = time. strftime("%Y-%m-%d %H:%M:%S")
    minute = int(time[-5: -3])
    time_temp = parse(time[: -6])
    if minute > 30:
        time_temp += dt.timedelta(hours=1)
    time = time_temp.strftime("%Y-%m-%d %H:%M:%S")
    return time


class get_forecast(object):
    def __init__(self, date, ft):
        self.date = date
        self.ft = ft

    def __call__(self, station_id):
        forecast_num, forecast_cat_list, forecast_sim = self.get_forecast_samples(station_id)
        return forecast_num, forecast_cat_list, forecast_sim


    def get_forecast_time_list(self, date):
        date = parse(date)     
        delta_h = dt.timedelta(hours=1)
        time_list = [(date+i*delta_h).strftime("%Y-%m-%d %H:%M") for i in range(72)]
        return time_list

    def get_forecast_df(self, station_id):
        date = parse(ph.end_date) +dt.timedelta(days=1)
        conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
        meo_df = pd.read_sql('select * from forecast_met_site where file_date = %(date)s', con=conn, params={'date': date})
        chem_df = pd.read_sql('select * from forecast_chem_site where file_date = %(date)s', con=conn, params={'date': date})
        conn.close()
        chem_df = chem_df[chem_df['model'] == 'Mix']
        chem_df = chem_df[chem_df['site_id'] == station_id]
        chem_df_list = []
        for ft in ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']:
            chem_df_ft = chem_df[chem_df['chem_type'] == ft]
            chem_df_ft = chem_df_ft[['site_id', 'forecast_time', 'chem_value']]
            chem_df_ft.columns = ['station_id', 'time', ft]
            chem_df_ft = chem_df_ft.reset_index(drop=True)
            chem_df_list.append(chem_df_ft)
        chem_df = pd.concat(chem_df_list, axis=0)
        chem_df = chem_df.groupby(level=0, axis=0).first()
        meo_df = meo_df[meo_df['site_id'] == station_id]
        meo_df_list = []
        for ft in ['temperature', 'rh', 'wind_we', 'wind_sn']:
            meo_df_ft = meo_df[meo_df['met_type'] == ft]
            meo_df_ft = meo_df_ft[['site_id', 'forecast_time', 'met_value']]
            meo_df_ft.columns = ['station_id', 'time', ft]
            meo_df_ft = meo_df_ft.reset_index(drop=True)
            meo_df_list.append(meo_df_ft)
        meo_df = pd.concat(meo_df_list, axis=1)
        meo_df = meo_df.groupby(level=0, axis=1).first()
        # meo_df = meo_df.drop([0,1,74], axis=1).T
        meo_df.columns = ['humidity','station_id','temperature','time','v1','v2']
        # limit max wind_speed to 15
        meo_df['wind_speed'] = list(map(lambda x,y: np.min([(x**2+y**2)**0.5,15.5]),
                                            meo_df['v1'], meo_df['v2']))
        meo_df['wind_direction'] = list(map(lambda x,y: cal_wind_direction(x,y),
                                            meo_df['v1'], meo_df['v2']))
        # limit max humidity to 100
        meo_df['humidity'] = list(map(lambda x: np.min([100,x])/100, meo_df['humidity']))
        meo_df['temperature'] = list(map(lambda x: x-273, meo_df['temperature']))
        # meo_df['time'] = self.get_forecast_time_list(self.date)
        meo_df['day_time'] = meo_df['time'].map(lambda x: x.hour)
        meo_df['week_day'] = meo_df['time'].map(lambda x:x.weekday())
        meo_df = meo_df[['wind_speed','wind_direction','temperature','humidity','day_time','week_day']]
        return meo_df, chem_df


    def get_forecast_samples(self, station_id):
        forecast_station_df, sim_station_df = self.get_forecast_df(station_id)
        forecast_num = forecast_station_df[['temperature', 'humidity', 'wind_speed']]
        forecast_cat = forecast_station_df[['wind_direction','day_time','week_day']]
        forecast_sim = sim_station_df[[self.ft]]
        forecast_num = np.array(forecast_num).reshape(72,-1)
        forecast_sim = np.array(forecast_sim).reshape(72,-1)
        forecast_cat_list = []
        for col in forecast_cat.columns:
            cat_array = np.array(forecast_cat[col]).reshape(1,72,-1)
            forecast_cat_list.append(cat_array)
        return forecast_num, forecast_cat_list, forecast_sim
    
class anhui_cities(object):
    city_list = ['滁州']
    station_dict = {
                    '滁州':['老年大学','人大宾馆','监测站'],
                    '合肥':['明珠广场', '三里街', '琥珀山庄', '董铺水库', '长江中路', '庐阳区', '瑶海区', '包河区', '滨湖新区', '高新区'],
                    '宿州':['监测楼', '一中', '火车站'],
                    '淮北':['监测站', '烈山区政府', '职业技术学院'],
                    '亳州':['污水处理厂', '三国揽胜宫'],
                    '阜阳':['市监测站', '开发区', '阜阳职业技术学院'],
                    '蚌埠':['工人疗养院', '百货大楼', '二水厂', '蚌埠学院', '淮上区政府', '高新区'],
                    '淮南':['潘集区政府', '师范学院', '谢家集区政府', '八公山区政府', '焦岗湖风景区管理处', '益益乳业工业园'],
                    '六安':['监测大楼', '皖西学院', '朝阳厂', '开发区'],
                    '马鞍山':['湖东路四小', '天平服装', '慈湖二小', '马钢动力厂', '市教育基地'],
                    '安庆':['环科院', '马山宾馆', '联富花园', '安庆大学'],
                    '芜湖':['监测站', '科创中心', '四水厂', '五七二零厂'],
                    '铜陵':['市第四中学', '市公路局', '市新民污水厂', '市第九中学', '车站新区', '市职教基地'],
                    '宣城':['鳌峰子站', '敬亭山子站', '开发区子站'],
                    '池州':['池州学院', '平天湖', '老干部局'],
                    '黄山':['黄山区政府5号楼', '延安路89号', '黄山东路89号']
                    }
    station_id_dict = {
                    '滁州':[2298, 2299, 2300],
                    '合肥':[1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279],
                    '宿州':[2304, 2305, 2306],
                    '淮北':[2282, 2283, 2284],
                    '亳州':[2311, 2312],
                    '阜阳':[2301, 2302, 2303],
                    '蚌埠':[2270, 2271, 2272, 2273, 2274, 2275],
                    '淮南':[2276, 2277, 2278, 2279, 2280, 2281],
                    '六安':[2307, 2308, 2309, 2310],
                    '马鞍山':[1798, 1799, 1800, 1801, 1802],
                    '安庆':[2291, 2292, 2293, 2294],
                    '芜湖':[1794,1795,1796,1797],
                    '铜陵':[2285, 2286, 2287, 2288, 2289, 2290],
                    '宣城':[2316, 2317, 2318],
                    '池州':[2313, 2314, 2315],
                    '黄山':[2295, 2296, 2297]
                       }
    def get_station_list_df(city_list,station_dict,station_id_dict):
        station_list = []
        station_id_list = []
        station_city_list = []
        for city in city_list:
            # print(station_dict[city])
            station_list += station_dict[city]
            station_id_list += station_id_dict[city]
            station_city_list += [city for _ in range(len(station_dict[city]))]    
        list_dict = {'city':station_city_list,'station':station_list,'station_id':station_id_list}    
        return pd.DataFrame(list_dict)      

    station_list_df = get_station_list_df(city_list,station_dict,station_id_dict)

              
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
            data_df['ma_'+str(ma_window)] = data_df[ft].rolling(window=ma_window).mean()
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
            ######################### change for different version of pandas #########################################
            # data_df['ema'+str(ema_window)] = pd.Series(pd.ewma(data_df[ft], span=ema_window,min_periods=ema_window))
            data_df['ema' + str(ema_window)] = pd.Series(
                data_df[ft].ewm(span=ema_window, min_periods=ema_window).mean())
            ##########################################################################################################
        return data_df


class create_anhui_predict_samples(anhui_cities, add_statistical_features):
    def __init__(self,date, ft, hours):
        self.date = parse(date).strftime('%Y%m%d')
        self.ft = ft
        self.days = int(hours/24)
        self.date_list = self.get_date_list(self.date, self.days)
        self.scaler_path = 'data/scaler/'
        self.scaler1 = self.read_scaler(self.ft,1)
        self.scaler2 = self.read_scaler(self.ft,2)
        self.scaler3 = self.read_scaler(self.ft,3)

    def __call__(self):
        print('ft:', self.ft)
        x_test = []
        delta_hour = dt.timedelta(hours=1)
        time_list = []
        for date in self.date_list:
            time_list_temp = [(parse(date)+i *delta_hour).strftime('%Y-%m-%d %H:%M:%S') for i in range(24)]
            time_list +=time_list_temp
        time_df = pd.DataFrame(time_list,columns=['time'])
        time_df = time_df.sort_values('time')
        time_df = time_df.reset_index(drop=True,)
        for city in self.city_list:
            aq_city, meo_city = self.get_anhui_samples(city, self.date_list)
            get_forecasting = get_forecast((parse(self.date)+dt.timedelta(days=1)).strftime('%Y-%m-%d'), self.ft)
            for station in self.station_dict[city]:
                station_id = int(self.station_list_df[(self.station_list_df['city']==city)&
                                                      (self.station_list_df['station']==station)]['station_id'])
                # print('station_id:', station_id)
                numeric_forecast, categorical_forecast_list, forecast_sim = get_forecasting(station_id)
                numeric_data, categorical_data_list = self.get_station_samples(aq_city,meo_city,station_id,self.ft,time_df)
                test_num = np.array(numeric_data)
                test_num_temp1 = self.scaler1.transform(test_num[:,:-7])
                test_num_temp2 = self.scaler2.transform(test_num[:,-7:])
                if self.ft == 'PM2.5':
                    test_fore_num = np.concatenate((test_num[:,:3],numeric_forecast),axis=-1)
                    test_fore_num = self.scaler1.transform(test_fore_num)
                    test_fore_num = test_fore_num[:,3:].reshape(1,-1,numeric_forecast.shape[-1])
                else:
                    test_fore_num = np.concatenate((test_num[:,:1],numeric_forecast),axis=-1)
                    test_fore_num = self.scaler1.transform(test_fore_num)
                    test_fore_num = test_fore_num[:,1:].reshape(1,-1,numeric_forecast.shape[-1])
                    
                test_num = np.concatenate((test_num_temp1,test_num_temp2),axis=-1)
                test_num = test_num.reshape(1,-1,test_num.shape[-1])
                test_cat = []
                for x in categorical_data_list:
                    test_cat.append(np.array(x).reshape(1,-1,1))
                y_test_initial = self.scaler3.transform(forecast_sim)
                x_test_temp = [test_num]+test_cat+[y_test_initial.reshape(1,-1,1), test_fore_num]+categorical_forecast_list
                if len(x_test):
                    for i in range(len(x_test_temp)):
                        x_test[i] = np.concatenate([x_test[i],x_test_temp[i]],axis=0)
                else:
                    x_test = x_test_temp
        return x_test
        
    def read_scaler(self,ft,num=1):    
        from sklearn.externals import joblib
#        ft_dict = {'pm2.5':'PM2.5',
#                   'pm10':'PM10',
#                   'o3':'O3',
#                   'no2':'NO2',
#                   'co':'CO',
#                   'so2':'SO2'}   
        num_dict = {1:'01',2:'02',3:'03'}
        scaler_path = self.scaler_path+'scaler'+num_dict[num]+'_'+ft
        min_max_scaler = joblib.load(scaler_path)
        return min_max_scaler


    def get_date_list(self,date, days):
        current_date = parse(date)
        delta_day = dt.timedelta(days=1)
        date_list = [current_date.strftime('%Y%m%d')]
        for i in range(1,days):
            temp_day = current_date - delta_day*i
            date_list.append(temp_day.strftime('%Y%m%d'))
        return date_list
        
    def get_anhui_samples(self,city, date_list):
        aq = pd.DataFrame([])
        meo = pd.DataFrame([])
        for date in date_list:
            # read aq_file
            conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
            aq_cz = pd.read_sql('select * from observation_chem_site where file_date = %(date)s', con=conn,
                             params={'date': date})
            conn.close()
            # aq_cz = pd.read_csv(aq_file_path, encoding='gbk')
            aq_cz.columns = ['id','station_id','file_date','chem_type','chem_unit','time','chem_value']
            aq_cz = aq_cz[['station_id','chem_type','chem_value','time']]
            aq_cz = aq_cz.replace('_',np.nan)
            aq_cz['chem_value'] = aq_cz['chem_value'].map(lambda x: float(x))
            aq_cz = fill_missing_values_numeric(aq_cz)
            aq_cz_list = []
            for id in [2298, 2299, 2300]:
                aq_cz_list.append(aq_cz[aq_cz['station_id'] == id])
            aq_cz = pd.concat(aq_cz_list, axis=0)
            aq_cz_list = []
            for ft in ['PM2.5', 'PM10', 'O3', 'CO', 'SO2','NO2']:
                aq_cz_ft = aq_cz[aq_cz['chem_type'] == ft]
                aq_cz_ft = aq_cz_ft[['station_id','time', 'chem_value']]
                aq_cz_ft.columns = ['station_id', 'time', ft]
                aq_cz_ft = aq_cz_ft.reset_index(drop=True)
                aq_cz_list.append(aq_cz_ft)
            aq_cz = pd.concat(aq_cz_list, axis=0)
            aq_cz = aq_cz.groupby(level=0, axis=0).first()
            aq_cz['day_time'] = aq_cz['time'].map(lambda x: x.hour)
            aq_cz['week_day'] = aq_cz['time'].map(lambda x: x.weekday())
            aq_cz['time'] = aq_cz['time'].map(lambda x: changeTimeStamp(x))
            if len(aq):
                aq = pd.concat([aq, aq_cz], axis=0)
            else:
                aq = aq_cz
            aq = aq.drop_duplicates(subset='time')

            conn = pymysql.connect(host=db.host, port=db.port, user=db.user, passwd=db.password, db=db.db)
            meo_cz = pd.read_sql('select * from observation_met_city where file_date = %(date)s', con=conn,
                             params={'date': date})
            conn.close()
            meo_cz.columns = ['id','city_id','file_date','met_type','met unit','time','met_value']
            meo_cz = meo_cz[['city_id','met_type','met_value','time']]
            meo_cz = meo_cz[meo_cz['city_id'] == 341100]
            meo_cz['time'] = meo_cz['time'].map(lambda x: changeTimeStamp(x))
            meo_cz_list = []
            for ft in ['temperature', 'rh', 'wd', 'wp', 'weather']:
                meo_cz_ft = meo_cz[meo_cz['met_type'] == ft]
                meo_cz_ft = meo_cz_ft[['city_id','time', 'met_value']]
                meo_cz_ft.columns = ['city_id','time', ft]
                meo_cz_ft = meo_cz_ft.reset_index(drop=True)
                meo_cz_list.append(meo_cz_ft)
            meo_cz = pd.concat(meo_cz_list, axis=1)
            meo_cz = meo_cz.groupby(level=0, axis=1).first()
            meo_cz['rh'] = meo_cz['rh'].map(lambda x: float(x)/100)  # 100)
            meo_cz['temperature'] = meo_cz['temperature'].map(lambda x: float(x))
            meo_cz['wd'] = meo_cz['wd'].map(lambda x: wind_encoder(x))
            meo_cz['weather'] = meo_cz['weather'].map(lambda x: weather_encoder(x))
            meo_cz['wp'] = meo_cz['wp'].map(lambda x: wind_speed_decoder(x))
            meo_cz = fill_missing_values_numeric(meo_cz)
            if len(meo):
                meo = pd.concat([meo, meo_cz], axis=0)
            else:
                meo = meo_cz
        meo = meo.drop_duplicates(subset='time')
        return aq.sort_values('time'), meo.sort_values('time')

    def get_station_samples(self,aq,meo,station,ft,time_df):
        aq_station = aq[aq['station_id'] == station]
        aq_station = pd.merge(time_df,aq_station,how='outer',on='time')
        aq_station = aq_station.drop_duplicates(subset='time')
        aq_station = aq_station.reset_index()
        aq_station = aq_station.set_index('time')
        current_date = parse(self.date) + dt.timedelta(hours=23)
        tomorrow = current_date.strftime('%Y-%m-%d %H:%M:%S')
        aq_station = aq_station[:tomorrow]
        meo = meo.reset_index()
        meo = meo.set_index('time')
        meo = meo[:tomorrow]
        aq_meo = pd.merge(aq_station, meo, how='outer',on='time')
        aq_meo = fill_missing_values_numeric(aq_meo)
        if ft == 'PM2.5':
            aq_meo = aq_meo[['station_id', ft, 'NO2', 'SO2', \
                             'temperature', 'rh', 'wd', \
                             'wp', 'weather', 'day_time', 'week_day']]
            numeric_data, categorical_data_list = \
                split_numeric_categorical(aq_meo, nums=[ft,'NO2','SO2'] + ['temperature', 'rh', 'wp'],
                                      cats=['wd', 'weather','day_time','week_day'])
        else:
            aq_meo = aq_meo[['station_id', ft, \
                             'temperature', 'rh', 'wd', \
                             'wp', 'weather', 'day_time', 'week_day']]
            numeric_data, categorical_data_list = \
                split_numeric_categorical(aq_meo, nums=[ft] + ['temperature', 'rh', 'wp'],
                                      cats=['wd', 'weather','day_time','week_day'])

        numeric_data = self.add_ema(numeric_data,ft, ema_window_list=[6,12,24])
        numeric_data = self.add_ma(numeric_data,ft, ma_window_list=[6,12,24])
        numeric_data = self.add_gradient(numeric_data,ft)
        numeric_data = numeric_data.ffill().bfill()
        return numeric_data, categorical_data_list
        
def split_numeric_categorical(df, nums, cats):

    # split the numeric and categorical data
    cols_num = []
    for n in nums:
        cols = [col for col in df.columns if n in col]
        cols_num = cols_num + cols
    numeric_data = df[cols_num]

    cols_cat = []
    categorical_data_list =[]
    for c in cats:
        cols = [col for col in df.columns if c in col]
        cols_cat = cols_cat + cols
        categorical_data_list.append(df[cols])
    return numeric_data, categorical_data_list
    

if __name__ == '__main__':
    date = ph.end_date  # the day before the local data
    fts = ['PM2.5', 'PM10', 'O3', "CO", "NO2", "SO2"]
    n = 72
    for ft in fts[2:3]:
        a = create_anhui_predict_samples(date, ft, n)
        x_test = a()








