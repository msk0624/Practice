import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19GenAgeCaseInfJson'
params ={'serviceKey' : 'wMkKn6Td+JzETi9VAIE4qRVAup7e3ozfdQPCwAWrMxP2FxJ6FCjGdwIdOtCywwgPK2nUlrGv8JjBX0tMzFLMCA==', 'pageNo' : '1', 'numOfRows' : '10', 'startCreateDt' : '20200310', 'endCreateDt' : '20200414' }

response = requests.get(url, params=params)
print(response.content)

#open api 어떻게 활용하는겨;;