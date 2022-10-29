import requests
import time
from pymongo import MongoClient

url = "https://goqual.io/openapi/homes"

response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
# 방찾는 url https://goqual.io/openapi/openapi/homes/${homeId}/rooms

homes_list = response.json()
for i in homes_list['result']:
    print(i)

    url = "https://goqual.io/openapi/homes/" + str(i['homeId']) + "/rooms"
    response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
    room_list = response.json()
    print(room_list)
