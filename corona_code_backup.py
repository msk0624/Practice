
#------------------------------------------------------------------------

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

#------------------------------------------------------------------------

#while True:
pressname = input("언론사 이름을 넣어주세요! : ")
for i in df_wordcloud['언론사'].unique():
    if pressname == i:
        wordstr = str(df_wordcloud.loc[df_wordcloud['언론사'] == i,'키워드'].tolist())
        wordlist = wordstr.split(",")

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

#혹시 모르니까 여기에 백업해논거