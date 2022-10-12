import requests  # 구역을 몽고 디비에 추가하는 파이썬 파일
import json
from pymongo import MongoClient
from ast import literal_eval

# 무위사 homeID :43946807
url = "https://goqual.io/openapi/homes"
# 이거는 그 헤이홈 Open API 가이드에서 상세설명 에서 원하는 데이터 마다 URL이 달라

response = requests.get(url, headers={'Authorization': 'Bearer 783ac103-e827-4694-8d69-e53b3a5f607b'})
# 인증토큰? 이렇게 넣으면 답이 와
print(response.json())
homes_list = response.json()
print(homes_list['result'][0]['name'])
