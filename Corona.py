import pandas as pd
import os

#print(os.getcwd())
#print(os.listdir())

pd.set_option("display.max_rows", None, "display.max_columns", None)
df_corona = pd.read_csv("./20210731_bigdata_corona.csv")
df_corona.drop(["주소","일자","원본주소"],axis=1,inplace=True)
#print(df_corona.head())
print(df_corona.columns)

#사건 사고 분류 nan값 엄청 많아서 자세히 보고 빼든지, 새로 값 추가 하든지 해야할듯?
#print(df_corona["사건_사고 분류1"].unique())
#print(df_corona["사건_사고 분류2"].unique())
#print(df_corona["사건_사고 분류3"].unique())
#일단 개많네..?


#음 일단 언론사, 기고자별로 분류 하는 거 해야 될 거 같고.
#통합 분류 1,2를 합치든지 뭘하든지 해서 Nan값 없애야 할 거 같고
