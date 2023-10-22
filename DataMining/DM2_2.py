import pandas as pd
import numpy as np

#=====================Q6
#6)
csv_file2 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/Mall_Customers.csv')
#print(csv_file2.head(10))

#=====================Q7
#7)
from sklearn.preprocessing import LabelEncoder

DF = pd.DataFrame(csv_file2)
le = LabelEncoder()
DF['Gender_OneHot'] = le.fit_transform(DF['Gender'])
# print(DF.head(10))

#=====================Q8_Q9
#8)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
x = DF[DF.columns[DF.columns.isin(['Gender_OneHot','Age','Annual Income (k$)', 'Spending Score (1-100)'])]]

principalComponents = pca.fit_transform(x)
col = ['pc1', 'pc2', 'pc3']

principalDf = pd.DataFrame(data = principalComponents, columns = col)
# print(principalDf.head(10))
# print("============")
# print(pca.explained_variance_ratio_)

