import requests

url = "https://api.shasta.trongrid.io/v1/accounts/address"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)
print(url)
print(response.text)
