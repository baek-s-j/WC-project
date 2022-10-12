from pymongo import MongoClient
from dateutil.relativedelta import relativedelta
import time
from datetime import datetime

now = time  # 5~7은 지금 현재시각을 가져오는 코드임 2022-08-22 20:48:57 이렇게

anytime = now.strftime('%Y-%m-%d')
print((datetime.now() - relativedelta(months=1)).strftime('%Y-%m-%d'))

client = MongoClient(
    'mongodb+srv://baeksj:1523@wcdb.e9pleut.mongodb.net/?retryWrites=true&w=majority')  # 여기서 부터는 몽고디비에 데이터 넣는것
db = client.wc_project
collection = db.thdata
min_time = list(collection.find().sort([("time", 1)]).limit(1))
max_time = list(collection.find().sort([("time", -1)]).limit(1))

print(min_time[0]["time"])
print(max_time[0]["time"])

# tmp = list(collection.find({"zone_name": "내부 후불벽화-5번-무위사", "time": {"$gte": "2022-01-01", "$lte": "2022-01-06"}})
#            )
# print(tmp)
