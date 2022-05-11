import os
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import collections
from wordcloud import WordCloud, STOPWORDS
import konlpy
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
okt.pos("안녕 내이름 혜성이")

df_corona = pd.read_csv("./20210731_bigdata_corona.csv")
df_corona.drop(["주소","원본주소"],axis=1,inplace=True)
df_corona.head()
df_corona.info()

