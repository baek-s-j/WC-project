import csv
import pymongo
import pandas as pd
import seaborn as sns
from pymongo import MongoClient

client = MongoClient('mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')
for k in range(0, 1):

    if k == 0:
        db = client.wc_project
    elif k == 1:
        db = client.wc_project_직지사
    else:
        db = client.wc_project_흥국사

    collection = db.zone
    z_result = list(collection.find())

    collection = db.thdata
    for i in z_result:
        resource_src = "C:/Users/qor27/PycharmProjects/WC_project/th_csv/" + i['name'] + "-ANSI.csv"
        tdf = pd.read_csv(resource_src, encoding='cp949')

        new_df = pd.DataFrame()
        new_df['time'] = tdf['업데이트 시간'].str.slice(start=0, stop=16)
        new_df['zone_name'] = tdf['센서']
        new_df['temperature'] = tdf['온도']
        new_df['humidity'] = tdf['습도']

        collection.insert_many(new_df.to_dict('records'))
