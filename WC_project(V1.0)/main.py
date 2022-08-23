import requests
import json

url = "https://ynr59py7db.execute-api.us-east-1.amazonaws.com/default/select_data?num=1"

abc = {
    "num": 5,
    "data_name":"bbakc"
}

response = requests.get(url)

print(response.text)
