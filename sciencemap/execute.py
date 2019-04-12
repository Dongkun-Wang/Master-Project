from py2neo import *
import pandas as pd
graph = Graph('http://localhost:7474', username='neo4j', password='9527')
df=graph.run('MATCH p=shortestPath((bacon:element {name:"电影院"})-[*]-(meg:element {name:"总统套房"}))RETURN p').to_data_frame()
print(df.iloc[0][0])