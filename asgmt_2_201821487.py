# 김혜성 / 201821487 / ghtn2638@ajou.ac.kr

# 0. library and read csv
import matplotlib.pyplot as plt
import pandas as pd
from importlib_metadata import NullFinder
from numpy import NaN, kron


# Q1. Generate texts with a couple of sentences that describe the data set based on your empirical insight and statistical approach. 
df = pd.read_csv('Salaries.csv')
print(df)
print(df.info())
print("The data set consists of the man's(or woman) pays that described various factors. But it has many null and meaningless values. So if I do something with this data set, I will need to preprocess data")

# Q2. What is the average of the variable “BasePay”? / 완
# if we check's df['BasePay'] information. we have 605 null results in dataframe. And there are some str objects("Not Provided").
# so I delete this null and str object because it is necessary to datapreprocessing. And then, I will attain BasePay average.
# but I will not remove '0' results of Basepay. because I think it is meaningful.
df['BasePay']
df_BasePay = df[(df['BasePay'] != "Not Provided") & (df['BasePay'].isnull() == False)]
df_BasePay['BasePay'] = df_BasePay['BasePay'].astype('float', errors = 'raise')
df_BasePay
df_BasePay.info()
# It is sucessful!
AvgDf_Basepay = sum(df_BasePay['BasePay']) / len(df_BasePay['BasePay'])
print('The average of Basepay is {0}.'.format(AvgDf_Basepay))


# Q3. What is the highest value of the variable “OvertimePay”? / 완
df['OvertimePay']
# Delete "Not provided" in OvertimePay because of data preprocessing.
df_OvertimePay = df[(df['OvertimePay'] != "Not Provided")]
df_OvertimePay
df_OvertimePay.info()
df_OvertimePay['OvertimePay'] = df_OvertimePay['OvertimePay'].astype('float', errors = 'raise')

MaxOvertimePay = max(df_OvertimePay['OvertimePay'])
print("The highest value of OvertimePay is {0}.".format(MaxOvertimePay))


# Q4. What is the job title of the observation, “JOSEPH DRISCOLL”? Tip: Use all capitalized
#     letters to search the name. Otherwise, your code may provide a different answer that
#     doesn’t match up the observation because the data set has another observation of “Joseph
#     Driscoll” with the first letters of each word capitalized. / 완
df.columns
# We can obeserve Job title of JOSEPH DRISCOLL by using EmployeeName filtering.
df.loc[df['EmployeeName'] == "JOSEPH DRISCOLL"]
JobT_JOS = ''.join(df.loc[df['EmployeeName'] == "JOSEPH DRISCOLL", 'JobTitle'].to_list())
print("His Job Title is {0}.".format(JobT_JOS))


# Q5. How much has “JOSEPH DRISCOLL” earned (including the variable “Benefits”)? / 완
#df[['BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df[df['EmployeeName'] == "JOSEPH DRISCOLL"][['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
# The value of TotalpayBenefits are money that employees earned included benefits.
Earn_JOS = str(df[df['EmployeeName'] == "JOSEPH DRISCOLL"]['TotalPayBenefits'].to_list())
Earn_JOS = float(Earn_JOS.replace("[","").replace("]",""))
Earn_JOS
print("He earned {}.".format(Earn_JOS))


# Q6. Who is the person paid the most (including benefits)? / 완
df[['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df.info()
#TotalPayBenefits have no Nan results. so we dont have to datapreprocess!
max(df['TotalPayBenefits'])
df[df['TotalPayBenefits'] == max(df['TotalPayBenefits'])]
MostP_Emp = ''.join(df[df['TotalPayBenefits'] == max(df['TotalPayBenefits'])]["EmployeeName"].tolist())
print("The person who is paid the most is {}.".format(MostP_Emp))


# Q7. Who is the person paid the least (including benefits)? Explain if you notice anything
#     strange about the salaries paid to the person with the lowest earnings? / 완
df[df['TotalPayBenefits'] == min(df['TotalPayBenefits'])]
LeastP_Emp = ''.join(df[df['TotalPayBenefits'] == min(df['TotalPayBenefits'])]["EmployeeName"].tolist())
print("The person who is paid the most is {}.".format(LeastP_Emp))

df[df['EmployeeName'] == "Joe Lopez"][['EmployeeName','BasePay','OvertimePay','OtherPay','Benefits','TotalPay','TotalPayBenefits']]
df[df['EmployeeName'] == "Joe Lopez"][['EmployeeName','OtherPay','TotalPayBenefits']]
Paid_Joe = ''.join(df[df['EmployeeName'] == "Joe Lopez"]['OtherPay'].to_list())
print("It is strange because her salaries paid is minus results. because the value of Otherpay is {} and the value of Others are 0.".format(Paid_Joe))


# Q8. How many unique job titles can we see in the data set? / 완
df['JobTitle']
df['JobTitle'].unique()
Unique_JobT = len(df['JobTitle'].unique()) - 1
# Results is necesarry to substract 1. because one of the values of JobTitle is "Not provided".
print("The counts of unique Job Titles are {}.".format(Unique_JobT))


# Q9. What are the top 5 job titles we can most frequently see in the data set? / 완
SortDf = df[['Id','JobTitle']].groupby('JobTitle').count()['Id'].sort_values(ascending=False)
SortDf
Sort = SortDf.head(5)
Sort = ", ".join(Sort.index.tolist())
print("The top 5 job titles that we can most frequently see in the data set are {}.".format(Sort))


# Q10. How many job titles were occupied by a single person only in 2013? / 완
df.columns
Single_JobT = df[df['Year'] == 2013].groupby('EmployeeName').count().sort_values(by="JobTitle", ascending=False)
Single_JobT
Single_JobT.info()
# We have to attain the number that job titles were occupied by a single person. 
Number_Sin = len(Single_JobT[Single_JobT['Id'] == 1])
print("The number that job titles were occupied by a single person in 2013 is {}.".format(Number_Sin))

# Q11. How many people have the word “Chief” in their job titles? / 완
df[df["JobTitle"].str.contains('Chief', case=True) == True]
Number_Chief = len(df[df["JobTitle"].str.contains('Chief', case=True) == True])
print("The number that people have the word(Chief) in their job titles is {}.".format(Number_Chief))


# Q12. Visualize a histogram to show the distribution of the variable “TotalPay” with all the
#      observations in the data set. / 완
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
#      the variable “TotalPay”: i.e., low, medium, high salaries.

# In order to analyze accurately, it is necessary to clarify the criteria.
# but In this graph, the criteria for salaries catagorhy(low, medium, high) were arbitrarily divided by me.
df["Totalpay_Cata"] = "Not Yet"
df["Totalpay_Cata"][df['TotalPay'] >= 90000] = "High"
df["Totalpay_Cata"][(df['TotalPay'] >= 10000) & (df['TotalPay'] <= 90000)] = "medium"
df["Totalpay_Cata"][df['TotalPay'] <= 10000 ] = "low"

df[['Id','Totalpay_Cata']].groupby(['Totalpay_Cata'],as_index=True).count().plot.bar()
plt.show()

