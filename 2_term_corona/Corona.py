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
from wordcloud import WordCloud, STOPWORDS

from matplotlib import font_manager, rc #밑에 4줄까지 plt.plot 한글화 패치
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

pd.set_option("display.max_rows", None, "display.max_columns", None) #출력 할때 잘리는 거 없앰

df_corona = pd.read_csv("./20210731_bigdata_corona.csv") #100mb 이상이어서 올릴려고 git lfs로 깃허브에 올림
df_corona.drop(["주소","원본주소"],axis=1,inplace=True) #일단 바로 쓸데 없는 열 삭제


#음 일단 언론사, 기고자별로 분류 하는 거 해야 될 거 같고.


df_sample = df_corona.iloc[0:1000, :]

#평균값 기준 이상값들만 표시 나머지는 others
#print(df_sample.head())
#----------------------------------------------------------------------------------------------

df_group = df_sample.iloc[:,[1,3]].groupby('언론사').count().reset_index()
df_group['제목'].mean() # 여기까지 이제 언론사 중앙값 구한겨

#df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
#print(df_sample['제목'].isnull().sum())

for i in df_group['제목'].sort_values(ascending=True):
    if i >= df_group['제목'].mean():
        mlist_dfg = df_group['언론사'][df_group['제목']==i]
        mlist_dfg = mlist_dfg.to_list()
        mlist_dfg.append(i)
        #a = "".join(a) # 만약 str형태로 해야되면
        break
    
df_bfmap = df_sample.iloc[:,[1,3]].groupby('언론사').count().sort_values(by='제목').reset_index()
df_bfmap.columns = ['언론사', '기사개수']


def map_df(x):
    others = []
    index_others = []    
    dict_bfmap = dict(zip(x.index.tolist(),x['기사개수'].tolist()))
    for i,(j,v) in enumerate(dict_bfmap.items()):
        if v < mlist_dfg[1]:
            others.append(v) #평균보다 낮은 기사개수들을 리스트에 추가
            index_others.append(j)
    others = sum(others) / len(others) * 2 #밸런스 패치를 위해 나머지 기사 개수를 '남은개수 평균 *2' 로 잡기
                                                 #리스트에 있는 기사 개수들을 sum하고 나눔
    
    for k in index_others:
        x['언론사'][k] = '기타'
        x['기사개수'][k] = others
    df_afmap = x.drop_duplicates() #중복 값 제거
    return df_afmap #평균 이하 값들을 기타로 매핑하는 작업 끝

sns.barplot(data=map_df(df_bfmap),
            x='언론사',
            y='기사개수')
plt.xticks(rotation = 70) #언론사 별 낸 기사 개수 barplot을 시각화

#----------------------------------------------------------------------------------------------

#print(plt.show())


#통합 분류 1,2를 합치든지 뭘하든지 해서 Nan값 없애야 할 거 같고
#사건 사고 분류 nan값 엄청 많아서 자세히 보고 빼든지, 새로 값 추가 하든지 해야할듯?
#키워드를 좀 분석해야 채울 수 있을 거 같음
#print(df_corona["사건_사고 분류1"].unique())
#print(df_corona["사건_사고 분류3"].unique())
#print(df_corona["사건_사고 분류2"].unique())


#언론사 별로 낸 기사의 키워드를 wordcloud화 할 수 있게 하고 싶다.
#그리고 그 후엔 전체 기사 키워드를 가능하게 끔 하는 것도...
df_wordcloud = df_sample.drop(["일자","기고자","제목","통합 분류1","통합 분류2","통합 분류3","사건_사고 분류1",
                               "사건_사고 분류2","사건_사고 분류3","개체명(인물)","개체명(지역)","개체명(기업기관)",
                               "특성추출","본문"],axis=1)
#print(df_wordcloud.head())

print("언론사 목록",df_wordcloud['언론사'].unique())
wordlist = []
while True:
    pressname = input("추출하고 싶은 언론사 이름을 넣어주세요!(전체 입력 가능) : ")
    if pressname == "전체":
        wordstr = str(df_wordcloud.iloc[:,1].tolist())
        wordlist = wordstr.split(",")
        print("전체를 추출합니다.")
        break
    else:    
        for i in df_wordcloud['언론사'].unique():
            if pressname == i:
                wordstr = str(df_wordcloud.loc[df_wordcloud['언론사'] == i,'키워드'].tolist())
                wordlist = wordstr.split(",")
        if pressname not in df_wordcloud['언론사'].unique().tolist():
            print("해당 언론사는 없습니다. 다시 입력해주세요.")
        addword = input("더 추춣라고 싶은 언론사가 있습니까?(0:아니오, 그외:예) ")
        if addword == str(0):break
        elif addword == str(1):continue
#      잘 돌아간다잉

wordcounts = {}
for word in wordlist:
    if word not in wordcounts:
        wordcounts[word] = 0

    wordcounts[word] += 1


wordcounts_n = dict()
for (key, value) in wordcounts.items():
       # 일정 기준치 이상의 개수만 가진 단어만 워드클라우드화
   if value > 10:
       wordcounts_n[key] = value

#print(wordcounts_n)
#얘를 나중에 def써서 함수로 만들면 됨.

#print(df_wordcloud.loc[df_wordcloud['언론사'] == '디지털타임스','키워드'].tolist())
#   ★★'KoNLPy' 라는 한글 형태소 분석 패키지가 있는데 이거는 키워드 말고 본문이나 제목 분석할때 도움될듯!★★
#   spwords = set(STOPWORDS)
#   spwords.add('') 이건 나중에 본문할때 해볼끼..?

wc = WordCloud(font_path = "C:/Windows/Fonts/NGULIM.TTF", max_font_size=200, 
                background_color='white', width=800, height=800).generate_from_frequencies(wordcounts_n)

plt.figure(figsize=(10, 8))
plt.imshow(wc)
plt.tight_layout(pad=0)
plt.axis('off')
#print(plt.show())

#   이제 워드클라우드화까진 되니까 함수로 전체도 선택할 수 있게 만들자!!