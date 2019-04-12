#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:24:26 2019

@author: chenxi
"""

import requests
import json
import re
from switch import Simplified2Traditional, Traditional2Simplified
import pandas as pd
import pymysql
from jay import get_jay_words
import time

##r = requests.get('http://shuyantech.com/api/cndbpedia/ment2ent?q={}'.format(name))
#r = requests.get('http://shuyantech.com/api/cndbpedia/avpair?q={}'.format(name))
#content = json.loads(r.content.decode())
#graph = content['ret']
#jay_path = '/data/chenxi/knowledge_graph/jay_clean/'
#name_list = ['篮球','足球']
#name_list = get_jay_words(jay_path)
#df = []

def download_from_conceptnet(name):
    name = Simplified2Traditional(name)
    print(name)
    obj = requests.get('http://api.conceptnet.io/c/zh/{}'.format(name)).json()
    nextpage = 1
    edges_list = [obj['edges']]
    while nextpage:
        try:
            nextpage = obj['view']['nextPage']
        except:
            nextpage = None     
        if nextpage:
            obj = requests.get('http://api.conceptnet.io{}'.format(nextpage)).json()
            edges_list.append(obj['edges'])   
    return edges_list

def get_3tuple(name):
    edges_list = download_from_conceptnet(name)
    start_list = []
    end_list = []
    rel_list = []
    weight_list = []
    for edges in edges_list:
        for i in range(len(edges)):
            relation = edges[i]['rel']['label']
            weight = edges[i]['weight']
            if relation == "HasA":
                start = edges[i]['end']['label']
                end = edges[i]['start']['label']
            else:
                start = edges[i]['start']['label']
                end = edges[i]['end']['label']

            if not edges[i]['surfaceText'] or \
                Traditional2Simplified(start) == Traditional2Simplified(end):
                    continue
            else:
                start_lang = edges[i]['start']['language']
                end_lang = edges[i]['end']['language']
            if start_lang == 'zh' and end_lang == 'zh':
                start_list.append(Traditional2Simplified(start))
                end_list.append(Traditional2Simplified(end))
                rel_list.append(relation)
                weight_list.append(weight)
    df_temp = pd.DataFrame({'start': start_list,
                            'end': end_list,
                            'relation': rel_list,
                            'weight': weight_list})
    return df_temp
#    df.append(df_temp)
#    time.sleep(2)
# df = pd.concat(df)

def transfer_to_sql(df):
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='science', charset='utf8')
    cur = conn.cursor()
    for i in range(len(df)):
        temp = df.iloc[i]
        cur.execute("INSERT INTO edges (start,finish,relation,weight) VALUES (%s, %s, %s, %s);",
                    (temp['start'],temp['end'],temp['relation'],float(temp['weight'])))
    conn.commit()
    cur.close()
    conn.close()

def get_one_link_words(jay_path):
    name_list = get_jay_words(jay_path)
    new_name_list = []
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='science', charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT finish FROM edges")
    rows = cur.fetchall()
    rows = list(set(rows))
    for i in range(len(rows)):
        if rows[i][0] not in name_list:
            new_name_list.append(rows[i][0])
    cur.close()
    conn.close()
    return new_name_list


if __name__ == '__main__':
    jay_path = 'jay_clean/'
    name_list = get_one_link_words(jay_path)
    # name_list = get_jay_words(jay_path)
    df = []
    count = 0
    for name in name_list:
        count += 1
        df_temp = get_3tuple(name)
        # time.sleep(1)
        df.append(df_temp)
        if count % 20 == 0:
            df = pd.concat(df)
            transfer_to_sql(df)
            df = []
        elif count == len(name_list):
            df = pd.concat(df)
            transfer_to_sql(df)
            df = []
        else:
            pass
