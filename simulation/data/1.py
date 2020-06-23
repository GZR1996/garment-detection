import requests

url = "http://"

payload = ""
headers = {
    'cache-control': "no-cache",
    'Postman-Token': "dd318de4-1ba0-48c6-b8f4-95d3258fefa6"
    }

response = requests.request("GET", url, data=payload, headers=headers)

print(response.text)