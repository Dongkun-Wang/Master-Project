
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta
import pymysql
from ftplib import FTP
import os
from cfg import *

met_types = ['wind_we', 'wind_sn', 'temperature', 'humidity', 'pressure', 'radiation', 'rh']
met_units = ['m/s', 'm/s', 'K', 'kg/kg', '100pa', 'W^2', '%']


def get_data_from_ftp(fd):
    ftp = FTP('14.116.179.204')
    ftp.login()
    ftp.retrbinary('RETR /ftp1/fcst/metsite_{}_12z.csv'.format(fd),
                   open('../data/fetch_data/metsite_data/metsite_{0}_12z.csv'.format(fd), 'wb').write)


def metsite_transform(file_date, n=72):
    """
    Transform FTP metsite files to database format
    :param file_date: local file path
    :param n: predict n future hours, default 72
    :return: a new dataframe
    """
    p = '../data/fetch_data/metsite_data/metsite_{}_12z.csv'.format(file_date)
    if not os.path.exists(p):
        get_data_from_ftp(file_date)
    metsite = pd.read_csv(p, header=None)
    metsite = metsite.iloc[:, :-1]
    metsite.columns = ['city_id', 'site_id'] + ['{}h'.format(i) for i in range(n)]
    new_metsite = pd.DataFrame([])
    for site_id in metsite['site_id'].unique():
        met_1site = metsite[metsite['site_id'] == site_id].reset_index().copy()
        met_1site['met_type'] = met_types
        met_1site['met_unit'] = met_units
        met_1site['file_date'] = [parse(file_date).strftime('%Y-%m-%d %H:%M:%S') for _ in range(met_1site.shape[0])]
        cons_cols = met_1site[['site_id', 'file_date', 'met_type', 'met_unit']]
        new_met_1site = pd.DataFrame([])
        for i in range(n):
            forecast_time = pd.DataFrame([(parse(file_date) + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for _ in
                                          range(met_1site.shape[0])], columns=['forecast_time'])
            new_met_1site_1h = pd.concat((cons_cols, forecast_time, met_1site[['{}h'.format(i)]]), axis=1)
            new_met_1site_1h.rename(columns={'{}h'.format(i): 'met_value'}, inplace=True)
            if new_met_1site.size == 0:
                new_met_1site = new_met_1site_1h
            else:
                new_met_1site = pd.concat((new_met_1site, new_met_1site_1h), ignore_index=True)
        if new_metsite.size == 0:
            new_metsite = new_met_1site
        else:
            new_metsite = pd.concat((new_metsite, new_met_1site), ignore_index=True)
    # print(new_metsite)
    return new_metsite


def values2db(file_date):
    """
    Parse and save metsite file to database
    :param file_path: local file path
    """
    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()
    metsite = metsite_transform(file_date)
    values = metsite.values
    for i in range(values.shape[0]):
        cursor.execute('INSERT INTO forecast_met_site(site_id, file_date, met_type, met_unit, forecast_time, met_value) '
                       'VALUES (%s, %s, %s, %s, %s, %s)', tuple(values[i]))
    con.commit()
    cursor.close()
    con.close()
    print('File of {}'.format(file_date) + ' done!')


if __name__ == '__main__':
    file_date = ph.start_date
    file_dates = [(parse(file_date) + timedelta(days=i)).strftime('%Y%m%d') for i in range(100)]
    for fd in file_dates:
        values2db(fd)











