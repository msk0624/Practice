from re import A
import pandas as pd
from geopy.geocoders import Nominatim

df = pd.read_excel('korean.xlsx', engine='openpyxl')
df['사업장소재지'].head()


geolocoder = Nominatim(user_agent = 'South Korea')

a = "대전광역시 서구 내금곡길"
def geocoding(address): 
    geo = geolocoder.geocode(address)
    crd = (geo.latitude, geo.longitude)
    print(crd)
    return crd

geocoding(a)