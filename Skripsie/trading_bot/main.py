import datetime as dt 
from luno_python.client import Client
import json
import requests
'''
The luno api uses a  client name and secret for Authentication
name -> Key ID 
Secret -> password
Because I am focusing on bitcoin it is the only pair I am going to use and test
'''
#constants and environment variables
key_id = 'pgdzch7f8x5p'
key_secret = 'BdAy8LI5W9jYCk_w_eORMO7bGUVLvXlsCaigVmDDAqg' 
url = 'https://api.luno.com/api/'
stream_url = 'wss://ws.luno.com/api/1/stream/'
PAIR = 'XBTZAR'


c = Client(api_key_id= key_id,api_key_secret= key_secret)
try:
    res = c.get_ticker(pair='XBTZAR')
    print(res)
except Exception as e :
    print(e)

with open('Historic_Price_Data_Test.txt','w') as f:
    
    
    try:
        f.writelines(
               str(c.get_candles(900,PAIR,1709090))
        )
    except Exception as e :
        print(e)
    
    f.close


#candles_json = json.loads(str(c.get_candles(900,PAIR,1709090)))
def proc_data(file):
    ch = ''
    candle_data = []
    lines = []
    tempStr = ''
    f = open(file,'r')
    ch = f.read(1)    
    while ch != '[':
        tempStr = tempStr + ch
        ch = f.read(1)
    Header = tempStr.strip('{').split(',')
    print(Header)
    tempStr = ' '
    print(ch)
    while ch != ']':
        ch = f.read(1)
        if ch == '{':
            tempStr = tempStr + ch
            while ch != '}':
                ch = f.read(1)
                tempStr = tempStr + ch
                
            candle_data.append(tempStr)
        print(tempStr)
        tempStr = ''

    for line in f:
        lines.append(line)

    f.close
    return lines
    
price_data = proc_data('Historic_Price_Data_Test.csv')

def file_test():
    f = open('test.txt','r')
    temp = ""
    while True:
        ch = f.read(1)
        if not ch:
            break
        print(ch)
        temp =  temp + ch

    print(temp)
    f.close


#file_test()
#get_candles()
#print(price_data)

