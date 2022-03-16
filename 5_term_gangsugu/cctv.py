import os
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import collections

df_cctv = pd.read_csv("./서울특별시강서구안심이CCTV설치현황.csv", encoding = 'utf-8')
df_cctv = df_cctv.drop(index=0, axis=0)
#print(df_cctv)

#Nan인 것과 아닌것으로 나눈후 merge하면 될듯
df_cctv_B1 = df_cctv[df_cctv['Unnamed: 7'].isna()==True] #얘는 MERGE용
df_cctv_B2 = df_cctv[df_cctv['Unnamed: 7'].isna()==False]

#안심주소를 for문으로 써서 하나씩 바꾸게 만들고, move_range로 범위이동하고 unnamed 열 삭제하면 됨.

df_cctv_B2["안심 주소"] = df_cctv_B2["안심 주소"] + df_cctv_B2["CCTV 용도"]
df_cctv_B2["CCTV 용도"] = df_cctv_B2["위도"]
df_cctv_B2["위도"] = df_cctv_B2["경도"]
df_cctv_B2["경도"] = df_cctv_B2["CCTV 수량"]
df_cctv_B2["CCTV 수량"] = df_cctv_B2["수정 일시"]
df_cctv_B2["수정 일시"] = df_cctv_B2["Unnamed: 7"]

df_cctv_B1.drop(["Unnamed: 7"],axis=1,inplace=True)
df_cctv_B2.drop(["Unnamed: 7"],axis=1,inplace=True)

df_cctv = pd.concat([df_cctv_B1,df_cctv_B2])
df_cctv.to_csv("cctvlocation.csv", encoding='utf-8-sig')



