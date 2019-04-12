from py2neo import Node, Relationship, Graph
import pymysql
import pandas as pd

conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='95279527', db='science', charset='utf8')
science = pd.read_sql('select * from edges', con=conn)
conn.close()
# print(science.iloc[1][1])
graph = Graph('http://localhost:7474', username='neo4j', password='9527')
for i in range(science.iloc[:,0].size):
    a = Node('element', name=science.iloc[i][0])
    b = Node('element', name=science.iloc[i][1])
    r = Relationship(a, science.iloc[i][2], b)
    r['weight'] = science.iloc[i][3]
    s = a | b | r
    graph.create(s)
