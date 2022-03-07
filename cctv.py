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

df_cctv[df_cctv['Unnamed: 7'].isna()==False]
#NaN 행이 아닌 것만 일단 뽑아오자