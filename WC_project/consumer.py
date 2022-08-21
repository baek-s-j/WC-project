import requests
import time
from pymongo import MongoClient

now = time
print(now.strftime('%Y-%m-%d %H:%M:%S'))
anytime=now.strftime('%Y-%m-%d %H:%M:%S')
url = "https://goqual.io/openapi/device/eb03df543e3a9b4d8fwshq"

response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})

print(response.text)
result = eval(response.text)
print(type(result))
result['time'] = anytime
print(result)
client = MongoClient('127.0.0.1', 27017)
db = client.wc_project
collection = db.thdata
collection.insert_one(result)
client.close()
