# -*- coding: utf-8 -*-

import pandas as pd
from dateutil.parser import parse
import requests
import io
import pymysql
from datetime import timedelta
import numpy as np
from cfg import ph

met_types = ['temperature', 'rh', 'precipitation', 'wd', 'wp', 'weather']
met_units = ['C', '%', 'mm', None, None, None]


def url2df(file_date):
    url = 'http://183.129.229.233:12345/' + 'meo_{}.csv'.format(file_date)
    html = requests.get(url).content
    meo = pd.read_csv(io.StringIO(html.decode('utf-8')))
    meo.rename(columns={'城市': 'city_name'}, inplace=True)
    meo[' 相对湿度'] = meo[' 相对湿度'].apply(lambda x: x[:-1])
    meo[' 风力'] = meo[' 风力'].apply(lambda x: x[:-1])

    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()

    # Get city_list
    cursor.execute('SELECT city_id, city_name FROM forecast_city_list')
    rows = cursor.fetchall()
    con.commit()
    cities = pd.DataFrame(list(rows), columns=['city_id', 'city_name'])
    cities['city_name'] = cities['city_name'].apply(lambda x: x.strip()[:-1])
    cursor.close()
    con.close()

    meo = pd.merge(meo, cities, on=['city_name'], how='inner')

    meo.rename(columns={'观测时间': 'observed_time', '气温': 'temperature', ' 相对湿度': 'rh', ' 降水量(mm)': 'precipitation', ' 风向': 'wd', ' 风力': 'wp', ' 天气 ': 'weather'}, inplace=True)
    meo = meo[['city_id', 'observed_time'] + met_types]

    # format meo data
    file_date_col = pd.DataFrame([parse(file_date).strftime('%Y-%m-%d %H:%M:%S') for _ in range(meo.shape[0])], columns=['file_date'])
    new_meo = pd.DataFrame([])
    for ix, met_type in enumerate(met_types):
        met_type_col = pd.DataFrame([met_type for _ in range(meo.shape[0])], columns=['met_type'])
        met_unit_col = pd.DataFrame([met_units[ix] for _ in range(meo.shape[0])], columns=['met_unit'])
        meo_1type = meo[['city_id', 'observed_time', met_type]].copy().reset_index(drop=True)
        meo_1type.rename(columns={met_type: 'met_value'}, inplace=True)
        meo_1type = pd.concat((meo_1type, file_date_col, met_type_col, met_unit_col), axis=1)
        if new_meo.size == 0:
            new_meo = meo_1type
        else:
            new_meo = pd.concat((new_meo, meo_1type), axis=0, ignore_index=True)

    new_meo = new_meo[['city_id', 'file_date', 'met_type', 'met_unit', 'observed_time', 'met_value']]

    new_meo['met_value'].replace({'_': None}, inplace=True)
    new_meo['met_value'].replace({np.nan: None}, inplace=True)
    new_meo['observed_time'] = new_meo['observed_time'].apply(lambda x: parse(x).strftime('%Y-%m-%d %H:%M:%S'))

    # print(new_meo)
    return new_meo


def values2db(file_date):

    """
    Parse and save metsite file to database
    :param file_path: local file path
    """
    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()

    meo = url2df(file_date)
    # print(meo.columns)
    values = meo.values
    for i in range(values.shape[0]):
        cursor.execute('INSERT INTO observation_met_city(city_id, file_date, met_type, met_unit, observed_time, met_value) VALUES (%s, %s, %s, %s, %s, %s)', tuple(values[i]))
    con.commit()

    cursor.close()
    con.close()

    print('File of {}'.format(file_date) + ' done!')


if __name__ == '__main__':
    file_date = ph.start_date
    file_dates = [(parse(file_date) + timedelta(days=i)).strftime('%Y%m%d') for i in range(100)]

    for fd in file_dates:
        # url2df(file_date)
        values2db(fd)
