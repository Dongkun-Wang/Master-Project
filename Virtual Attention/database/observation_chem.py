# -*- coding: utf-8 -*-

import pandas as pd
from dateutil.parser import parse
import requests
import io
import pymysql
from datetime import timedelta
from cfg import ph

chem_types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
chem_units = ['ug/m^3', 'ug/m^3', 'ug/m^3', 'ug/m^3', 'mg/m^3', 'ug/m^3']


def url2df(file_date):
    url = 'http://183.129.229.233:12345/' + 'aq_{}.csv'.format(file_date)
    html = requests.get(url).content
    air = pd.read_csv(io.StringIO(html.decode('utf-8')))
    air.rename(columns={'站点名': 'site_name', '城市': 'city_name'}, inplace=True)

    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()

    # Get site_list and city_list
    cursor.execute('SELECT city_id, site_id, site_name FROM forecast_site_list')
    rows = cursor.fetchall()
    con.commit()
    sites = pd.DataFrame(list(rows), columns=['city_id', 'site_id', 'site_name'])

    cursor.execute('SELECT city_id, city_name FROM forecast_city_list')
    rows = cursor.fetchall()
    con.commit()
    cities = pd.DataFrame(list(rows), columns=['city_id', 'city_name'])
    cities['city_name'] = cities['city_name'].apply(lambda x: x.strip()[:-1])
    cursor.close()
    con.close()

    sites = pd.merge(sites, cities, on='city_id', how='left')

    air = pd.merge(air, sites, on=['city_name', 'site_name'], how='inner')
    air.rename(columns={'观测时间': 'observed_time', 'pm2.5': 'PM2.5', 'pm10': 'PM10', 'o3': 'O3', 'no2': 'NO2', 'co': 'CO',
                        'so2': 'SO2'}, inplace=True)
    air = air[['site_id', 'observed_time'] + chem_types]

    # format air data
    file_date_col = pd.DataFrame([parse(file_date).strftime('%Y-%m-%d %H:%M:%S') for _ in range(air.shape[0])],
                                 columns=['file_date'])
    new_air = pd.DataFrame([])
    for ix, chem_type in enumerate(chem_types):
        chem_type_col = pd.DataFrame([chem_type for _ in range(air.shape[0])], columns=['chem_type'])
        chem_unit_col = pd.DataFrame([chem_units[ix] for _ in range(air.shape[0])], columns=['chem_unit'])
        air_1type = air[['site_id', 'observed_time', chem_type]].copy().reset_index(drop=True)
        air_1type.rename(columns={chem_type: 'chem_value'}, inplace=True)
        air_1type = pd.concat((air_1type, file_date_col, chem_type_col, chem_unit_col), axis=1)
        if new_air.size == 0:
            new_air = air_1type
        else:
            new_air = pd.concat((new_air, air_1type), axis=0, ignore_index=True)

    new_air = new_air[['site_id', 'file_date', 'chem_type', 'chem_unit', 'observed_time', 'chem_value']]

    new_air['chem_value'].replace({'_': None, '—': None}, inplace=True)

    return new_air


def values2db(file_date):

    """
    Parse and save metsite file to database
    :param file_path: local file path
    """
    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()
    air = url2df(file_date)
    values = air.values
    for i in range(values.shape[0]):
        cursor.execute('INSERT INTO observation_chem_site(site_id, file_date, chem_type, chem_unit, observed_time, chem_value) VALUES (%s, %s, %s, %s, %s, %s)', tuple(values[i]))
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













