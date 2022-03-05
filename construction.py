import requests
import pandas as pd
from pandas.io.json import json_normalize
from pandas import DataFrame
import pprint
import json

url = 'http://data.sisul.or.kr/AutoAPI/service/OpenDB/ConstructInfo/getConstructInfoQry'
params ={'serviceKey' : 'wMkKn6Td+JzETi9VAIE4qRVAup7e3ozfdQPCwAWrMxP2FxJ6FCjGdwIdOtCywwgPK2nUlrGv8JjBX0tMzFLMCA==', 'numOfRows' : '10', 'pageNo' : '1', 'pcnrtname' : '공사' }
#왜...서비스키가 안되지..?


response = requests.get(url, params=params)
content = response.text
content = content.replace('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>','')
#<?xml version="1.0" encoding="UTF-8" standalone="yes"?> 
import xmltodict, json
obj = xmltodict.parse(content)
#json_ob = json.loads(content) #json으로 안바뀌는 큰 이유중 하나는 xml형태여서임. xml 형태는 <>인데 json으로해서 dict타입으로 바꾸려면 {}형태가 필요.

content_obj = json.dumps(obj)

json_ob = json.loads(content_obj)
print(json_ob)
#print(type(json_ob))

#body = json_ob['response']['body']['items']['item'] #이 속에 원하는 정보가 담겨있음
#print(body)

# Dataframe으로 만들기
#dataframe = json_normalize(body)
#dataframe.to_csv("view.csv", encoding="utf-8-sig") #csv형태로 보기만하면된다...

#pp = pprint.PrettyPrinter(indent=4)
#print(pp.pprint(content))