
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta
import pymysql
from ftplib import FTP
import os
from cfg import *

chem_types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
chem_units = ['ug/m^3', 'ug/m^3', 'ug/m^3', 'ug/m^3', 'mg/m^3', 'ug/m^3']


def get_data_from_ftp(fd):
    ftp = FTP('14.116.179.204')
    ftp.login()
    ftp.retrbinary('RETR /ftp1/fcst/chemsite_{}_12z.csv'.format(fd),
                   open('../data/fetch_data/chemsite_data/chemsite_{0}_12z.csv'.format(fd), 'wb').write)


def chemsite_transform(file_date, n=72):

    """
    Transform FTP chemsite files to database format
    :param file_path: local file path
    :param n: predict n future hours, default 72
    :return: a new dataframe
    """
    p = '../data/fetch_data/chemsite_data/chemsite_{}_12z.csv'.format(file_date)
    if not os.path.exists(p):
        get_data_from_ftp(file_date)
    chemsite = pd.read_csv(p, header=None)
    # print(chemsite)
    chemsite = chemsite.iloc[:, :-1]
    chemsite.columns = ['city_id', 'site_id'] + ['{}h'.format(i) for i in range(72)]
    for col in ['city_id', 'site_id']:
        chemsite[col] = chemsite[col].astype(int)

    new_chemsite = pd.DataFrame([])
    for site_id in chemsite['site_id'].unique():
        chem_1site = chemsite[chemsite['site_id'] == site_id].reset_index().copy()
        chem_1site['chem_type'] = chem_types
        chem_1site['chem_unit'] = chem_units
        chem_1site['file_date'] = [parse(file_date).strftime('%Y-%m-%d %H:%M:%S') for _ in range(chem_1site.shape[0])]

        cons_cols = chem_1site[['site_id', 'file_date', 'chem_type', 'chem_unit']]
        new_chem_1site = pd.DataFrame([])
        for i in range(n):
            forecast_time = pd.DataFrame([(parse(file_date) + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for _ in
                                          range(chem_1site.shape[0])], columns=['forecast_time'])
            new_chem_1site_1h = pd.concat((cons_cols, forecast_time, chem_1site[['{}h'.format(i)]]), axis=1)
            new_chem_1site_1h.rename(columns={'{}h'.format(i): 'chem_value'}, inplace=True)
            if new_chem_1site.size == 0:
                new_chem_1site = new_chem_1site_1h
            else:
                new_chem_1site = pd.concat((new_chem_1site, new_chem_1site_1h), ignore_index=True)

        if new_chemsite.size == 0:
            new_chemsite = new_chem_1site
        else:
            new_chemsite = pd.concat((new_chemsite, new_chem_1site), ignore_index=True)
    # Specify the model
    new_chemsite['model'] = pd.Series(['Mix' for _ in range(new_chemsite.shape[0])])
    # print(new_chemsite)
    return new_chemsite


def values2db(file_date):

    """
    Parse and save metsite file to database
    :param file_path: local file path
    """
    con = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='database', charset='utf8')
    cursor = con.cursor()
    chemsite = chemsite_transform(file_date)
    values = chemsite.values
    for i in range(values.shape[0]):
        cursor.execute\
            ('INSERT INTO forecast_chem_site(site_id, file_date, chem_type, chem_unit, forecast_time, chem_value, model) VALUES (%s, %s, %s, %s, %s, %s, %s)', tuple(values[i]))
    con.commit()
    cursor.close()
    con.close()
    print('File of {}'.format(file_date) + ' done!')


if __name__ == '__main__':
    file_date = ph.start_date
    file_dates = [(parse(file_date) + timedelta(days=i)).strftime('%Y%m%d') for i in range(100)]
    for fd in file_dates:
        values2db(fd)


