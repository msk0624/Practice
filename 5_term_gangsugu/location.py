import os
from keras.engine import sequential
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import collections

df_location = pd.read_csv("./서울시_건설공사_추진_현황.csv", encoding='utf-8')

df_location.drop(["계획공정율","실적공정율","대비율","기준일자","웹카메라 RTSP 주소"],axis=1,inplace=True) #일부 열 삭제
#print(df_location.columns)

#print(df_location["공사위치"]) #여기에서 서울 강서구만 가져오면 됨
#print(df_location['공사위치'].str.contains('강서').astype('bool'))

df_gangsogu = df_location[(df_location['공사위치'].str.contains('강서').astype('bool'))]
#print(df_gangsogu)


#df_gangsogu.to_csv('강서구_공사위치.csv', encoding='utf-8-sig', index=False)