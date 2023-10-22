import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#=====================Q1
# csv_file = pd.read_csv('C:/Users/Lenovo/Desktop/Social_Network_Ads.csv')
# DF = pd.DataFrame(csv_file)
#
# DF['Age'].plot(kind='box', title='Ages of Social_Network_Ads')
# plt.show()
#
# #=====================Q2
# import seaborn as sns
#
# csv_file = pd.read_csv('C:/Users/Lenovo/Desktop/Mall_Customers.csv')
# DF = pd.DataFrame(csv_file)
#
# sns.jointplot(data=DF ,x='Gender', y='Annual Income (k$)',kind='scatter')
# plt.show()

#=====================Q3
from apriori_python import apriori

itemSetList = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

freqItemSet,rules = apriori(itemSetList, minSup=0.6, minConf=0.7)
print(freqItemSet)
print('#=============')
print(rules)
