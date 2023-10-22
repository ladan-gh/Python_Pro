import pandas as pd
import numpy as np

#=====================Q1
#1)
csv_file = pd.read_csv('C:/Users/Ladan_Gh/Desktop/Social_Network_Ads.csv')
df = pd.DataFrame(csv_file)
# print(df.head(10))

# =====================Q2_1
# 2)without sklearn
df = pd.DataFrame(csv_file)

df['EstimatedSalary_Normal'] = (df['EstimatedSalary'] - df['EstimatedSalary'].mean()) / (df['EstimatedSalary'].std())
df['Age_Normal'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std())

# print(df.head(10))

#=====================Q2_2
#2)with sklearn
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df[['Age_Zscore', 'EstimatedSalary_Zscore']] = sc.fit_transform(df[['Age', 'EstimatedSalary']])

#print(df[['Age_Zscore', 'EstimatedSalary_Zscore']].head(10))

#=====================Q3
# 3)
from sklearn.preprocessing import MinMaxScaler

dataset = np.array(df[['Age_Zscore', 'EstimatedSalary_Zscore']])
scaler = MinMaxScaler(feature_range=(0, 10))

#scaler.fit(dataset)
df[['Age_MinMax', 'EstimatedSalary_MinMax']] = scaler.fit_transform(dataset)

#print(df[['Age_MinMax', 'EstimatedSalary_MinMax']].head(10))

#=====================Q4
#4)
from sklearn.preprocessing import RobustScaler

data = df[['Age', 'EstimatedSalary']]
rb = RobustScaler()
df[['Age_RobustScaler', 'EstimatedSalary_RobustScaler']] = rb.fit_transform(data)

# print(df[['Age_RobustScaler', 'EstimatedSalary_RobustScaler']].head(10))

#=====================Q5
#5)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# print(df.head(10))

print("#==================")

reverse = df['Gender_OneHot'].head(10)
# print(le.inverse_transform(reverse))