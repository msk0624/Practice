from importlib_metadata import NullFinder
from numpy import NaN, kron


# 김혜성 / 201821487 / ghtn2638@ajou.ac.kr

# 0. library and read csv
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Salaries.csv')
df
df.info()

# Q1. Generate texts with a couple of sentences that describe the data set based on your empirical insight and statistical approach. (4 points)
#    (모든 답에 대햐여 경험적 통찰력과 통계적 접근법에 따라 데이터 세트를 설명하는 몇 개의 문장으로 텍스트를 생성하라.)


# Q2. What is the average of the variable “BasePay”?
# if we check's df['BasePay'] information. we have 605 null results in dataframe. And there is some str object in column.
# so we delete this null and str object and then, attain BasePay average.
# I don't remove '0' results. because i think it is meaningful.

df['BasePay']
df_BasePay = df[(df['BasePay'] != "Not Provided") & (df['BasePay'].isnull() == False)]
# df_BasePay = df[df['BasePay'].isnull() == False]
df_BasePay['BasePay'] = df_BasePay['BasePay'].astype('float', errors = 'raise')
df_BasePay.info()

sum(df_BasePay['BasePay']) / len(df_BasePay['BasePay'])
# A2. BasePay's average is 66325.44 을 문장으로

# Q3. What is the highest value of the variable “OvertimePay”?
df['OvertimePay']
df_OvertimePay = df[(df['OvertimePay'] != "Not Provided")]
df_OvertimePay['OvertimePay'] = df_OvertimePay['OvertimePay'].astype('float', errors = 'raise')
df_OvertimePay.info()
max(df_OvertimePay['OvertimePay'])
# A3. OvertimePay's highest value is 245131.88 을 문장으로

# Q4. What is the job title of the observation, “JOSEPH DRISCOLL”? Tip: Use all capitalized
#     letters to search the name. Otherwise, your code may provide a different answer that
#     doesn’t match up the observation because the data set has another observation of “Joseph
#     Driscoll” with the first letters of each word capitalized. 
df.columns
df.loc[df['EmployeeName'] == "JOSEPH DRISCOLL", 'JobTitle'].to_list()

# Q5. How much has “JOSEPH DRISCOLL” earned (including the variable “Benefits”)?
#df[['BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df[df['EmployeeName'] == "JOSEPH DRISCOLL"][['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df[df['EmployeeName'] == "JOSEPH DRISCOLL"]['TotalPayBenefits'].to_list()

# Q6. Who is the person paid the most (including benefits)?
df[['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df.info()
max(df['TotalPayBenefits'])
df[df['TotalPayBenefits'] == max(df['TotalPayBenefits'])]

# Q7. Who is the person paid the least (including benefits)? Explain if you notice anything
#     strange about the salaries paid to the person with the lowest earnings? 
df[df['TotalPayBenefits'] == min(df['TotalPayBenefits'])]
df[df['EmployeeName'] == "Joe Lopez"][['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
# OtherPay 부분이 마이너스임

# Q8. How many unique job titles can we see in the data set? 
df['JobTitle'].unique()
len(df['JobTitle'].unique()) - 1

# Q9. What are the top 5 job titles we can most frequently see in the data set?
a = df[['Id','JobTitle']].groupby('JobTitle').count()['Id'].sort_values(ascending=False)
b = a.head(5)
b.index.tolist()

# Q10. How many job titles were occupied by a single person only in 2013?
df.info()
df.columns
a = df[df['Year'] == 2013].groupby('EmployeeName').count().sort_values(by="JobTitle", ascending=False)
a.info()
len(a[a['Id'] == 1])

# Q11. How many people have the word “Chief” in their job titles?
df[df["JobTitle"].str.contains('Chief', case=True) == True]
len(df[df["JobTitle"].str.contains('Chief', case=True) == True])

# Q12. Visualize a histogram to show the distribution of the variable “TotalPay” with all the
#      observations in the data set. (3 points)
df["TotalPay"]
plt.hist(df["TotalPay"])
plt.show()

# Q13. Visualize a line chart where the x-axis indicates the year(over time) whereas the y-axis 
#      indicates the average “TotalPay” of all the observations.

df['Year'].unique()

def mean(x):
    y = sum(x) / len(x)
    return y

mean(df['TotalPay'][df['Year'] == 2011])

Average_TotalPay = []
for i in df['Year'].unique():
    z = mean(df['TotalPay'][df['Year'] == i])
    Average_TotalPay.append(float(z))
Average_TotalPay

plt.plot(df['Year'].unique(),Average_TotalPay)
plt.xlabel("The Year(over time)")
plt.ylabel("Total Pay of Average")
plt.show()

# Q14. Visualize a bar chart with the counts of observations across the three different ranges of
#      the variable “TotalPay”: i.e., low, medium, high salaries. (3 points)

df["Totalpay_Cata"] = "Not Yet"
df["Totalpay_Cata"][df['TotalPay'] >= 20000] = "High"
df["Totalpay_Cata"][(df['TotalPay'] >= 10000) & (df['TotalPay'] <= 20000)] = "medium"
df["Totalpay_Cata"][df['TotalPay'] <= 10000 ] = "low"

df[['Id','Totalpay_Cata']].groupby(['Totalpay_Cata'],as_index=True).count().plot.bar()
plt.show()
