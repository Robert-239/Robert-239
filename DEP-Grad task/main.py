import os
import requests
import pandas as pd 
import base58
from tqdm import tqdm
import json
from dotenv import load_dotenv

schema = {
    'walletAddress': str,
    'tokenID' : str,
    'tokenName' : str,
    'tokenABBV' : str,
    'balance' : float,
}


import cryptoapis as cpi
from requests.api import request

###
load_dotenv()
###
TOKEN_ID = 'TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t'
ASSET_ID = 'TRX_USDT_S2UZ'

try:
    df = pd.read_csv("Grad Task - data.csv").reset_index().rename_axis(None , axis = 1)
    print("File loaded")
except FileNotFoundError:
    print("File not found")

df_copy = df[df['balance'] > 0].copy()

print(df.info())
print(df_copy.info())
print(df_copy.head(1))
####################################################
# Tron main net URL

main_url = 'https://api.trongrid.io/'

# Tron Shasta test net

shasta_url = "https://api.shasta.trongrid.io/"

# Tron Nile test net

nile_url = "https://nile.trongrid.io/"
####################################################

try:
    tg_key = os.environ["TG_API_KEY"]
except ValueError:
    print("Key not found")

def info_by_adress(addr : str) -> str:
    
    url = f"https://api.trongrid.io/v1/accounts/{addr}"

    print(f"{url}\n")
    print(f"{addr}\n")

    payload = {
        "address" : addr ,
        "vissible" : True
    }

    headers ={
        "accept" : "application/json",
        #'TRON-PRO-API-KEY': tg_key
    }

    response = requests.get(url , headers = headers)

    return json.dumps(json.loads(response.text),indent= 4)


def validate_address(addr : str)-> str:
    url = f"https://api.trongrid.io/wallet/validateaddress"

    print(addr)

    payload = {
        "address" : addr ,
        "vissible" : True
    }

    headers ={
        "accept" : "application/json",
        "content-type" : "application/json",
        #'TRON-PRO-API-KEY': tg_key
    }

    response = requests.post(url ,json= payload, headers = headers)

    return json.dumps(json.loads(response.text),indent= 4)

#base58.b58decode_check(addr).hex().upper()

def getAccount(addr : str)-> str:
    url = f"https://api.trongrid.io/wallet/getaccount"

    payload = {
        "address" :  base58.b58decode_check(addr).hex().upper(),
        "visible" : False
    }

    print(payload['address'])

    headers ={
        "accept" : "application/json",
        "content-type" : "application/json",
        #'TRON-PRO-API-KEY': tg_key
    }

    response = requests.post(url,json = payload , headers = headers)

    return json.dumps(json.loads(response.text),indent= 4)


def getAccountBallance(addr : str) -> str:
    
    url = f"https://api.trongrid.io/wallet/getaccountballance"

    payload = {
        "account-identifier" : { "address" : addr},
        "block_identifier" : {
        "hash" : "0000000000010c4a732d1e215e87466271e425c86945783c3d3f122bfa5affd9",
        "number" : 68682 ,
        },
        "visible" : True
    }

    headers = {
        "accept" : "application/json",
        "content-type" : "application/json",
        }

    response = requests.get(url , json= payload , headers= headers)

    return json.dumps(json.loads(response.text),indent= 4)


def getContracts(addr :str ) -> str:
    url = f"https://api.trongrid.io/v1/accounts/{addr}/transactions/trc20?limit=20"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    
    return json.dumps(json.loads(response.text),indent= 4)




schema = {
    'walletAddress': str,
    'tokenID' : str,
    'tokenName' : str,
    'tokenABBV' : str,
    'balance' : float,
}






def walletTokenOverview(addr : str) -> dict:

    url = f"https://apilist.tronscanapi.com/api/account/token_asset_overview?address={addr}"
    
    headers = {
        "TRON-PRO-API-KEY" : "6ca01e78-34d7-44f1-97ea-a038467cf965"

    }
    dct = schema

    response = requests.get(url, headers= headers)

    json_data = json.loads(response.text)

    dct['walletAddress']= addr
    dct['tokenID'] = ''
    dct['tokenName'] = ''
    dct['tokenABBV'] =''
    dct['balance'] = 0.0

    #print(json_data)
    for token in json_data['data']:

        if (token['tokenId'] == TOKEN_ID):
            dct['tokenID'] = token['tokenId']
            dct['tokenName'] = token['tokenName']
            dct['tokenABBV'] = token['tokenAbbr']
            dct['balance'] = float(token['balance']) / pow(10,token['tokenDecimal'])

    

    
    return dct

def getTRC20info(addr : str) -> str:
    url = f"https://apilist.tronscanapi.com/api/transfer/trc20?address={addr}&trc20Id=TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t&start=0&limit=2&direction=0&reverse=true&db_version=1&start_timestamp=&end_timestamp="

    headers = {"accept": "application/json"}

    response = requests.get(url)
    
    return json.dumps(json.loads(response.text),indent= 4)



def testTronscan(addr : str)-> str:
    
    url = f"https://apilist.tronscanapi.com/api/account/wallet?address={addr}&asset_type=1"
    
    headers = {
        "TRON-PRO-API-KEY" : "6ca01e78-34d7-44f1-97ea-a038467cf965"

    }

    response = requests.get(url)

    dict = json.loads(response.text)


    return json.dumps(json.loads(response.text),indent=2)

print("\n\n")


def getCurrentWalletBallences():

    new_df = pd.DataFrame()
    adr = []
    index = 0
    for address in tqdm(df['address'], leave= True):
        out_dict = walletTokenOverview(address)
        temp_frame = pd.DataFrame(out_dict,index=[0])
        new_df = new_df._append(out_dict, ignore_index= True)
        #new_df= pd.concat([temp_frame,new_df.iloc[index]]).reset_index(drop=True)
        out_dict.clear()
        index+= 1

    print(new_df.info())

    new_df.to_excel("out_final.xlsx")

getCurrentWalletBallences()



#print(testTronscan(df_copy.head(1)['address'].to_string(index= False)))
#print(getTRC20info(df_copy.head(1)['address'].to_string(index= False)))
## contract address:     TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t


