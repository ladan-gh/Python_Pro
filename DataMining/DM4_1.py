import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#================text to csv
file1 = pd.read_csv("C:/Users/Ladan_Gh/Desktop/clustering/spiral.txt", delimiter = '	', header = None)

file1.columns = ['X', 'Y', 'Cluster_Num']
file1.to_csv('C:/Users/Ladan_Gh/Desktop/clustering/spiral.csv',index = None)

file2 = pd.read_csv("C:/Users/Ladan_Gh/Desktop/clustering/jain.txt", delimiter = '	', header = None)
file2.columns = ['X', 'Y', 'Cluster_Num']
file2.to_csv('C:/Users/Ladan_Gh/Desktop/clustering/jain.csv',index = None)

file3 = pd.read_csv("C:/Users/Ladan_Gh/Desktop/clustering/flame.txt", delimiter = '	', header = None)
file3.columns = ['X', 'Y', 'Cluster_Num']
file3.to_csv('C:/Users/Ladan_Gh/Desktop/clustering/flame.csv',index = None)

file4 = pd.read_csv("C:/Users/Ladan_Gh/Desktop/clustering/Aggregation.txt", delimiter = '	', header = None)
file4.columns = ['X', 'Y', 'Cluster_Num']
file4.to_csv('C:/Users/Ladan_Gh/Desktop/clustering/Aggregation.csv',index = None)

#================Q1
csv_file1 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/spiral.csv')
DF = pd.DataFrame(csv_file1)

sns.jointplot(data=DF ,x='X', y='Y',kind='scatter', color='purple')
plt.show()

#****************
csv_file2 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/jain.csv')
DF = pd.DataFrame(csv_file2)

sns.jointplot(data=DF ,x='X', y='Y',kind='scatter', color='blue')
plt.show()

#****************
csv_file3 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/flame.csv')
DF = pd.DataFrame(csv_file3)

sns.jointplot(data=DF ,x='X', y='Y',kind='scatter', color='green')
plt.show()

#****************
csv_file4 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/Aggregation.csv')
DF = pd.DataFrame(csv_file4)

sns.jointplot(data=DF ,x='X', y='Y',kind='scatter', color='yellow')
plt.show()