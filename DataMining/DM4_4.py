import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#================Q4
from sklearn.cluster import DBSCAN

csv_file1 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/spiral.csv')
DF = pd.DataFrame(csv_file1)


x = DF[["X","Y"]].values

dbscan = DBSCAN(eps = 0.5, min_samples = 4).fit(x)
labels = dbscan.labels_


plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#***************
# csv_file2 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/jain.csv')
# DF = pd.DataFrame(csv_file2)
#
#
# x = DF[["X","Y"]].values
#
# dbscan = DBSCAN(eps = 0.5, min_samples = 4).fit(x)
# labels = dbscan.labels_
#
#
# plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") # plotting the clusters
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
#
# #***************"""
# csv_file3 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/flame.csv')
# DF = pd.DataFrame(csv_file3)
#
#
# x = DF[["X","Y"]].values
#
# dbscan = DBSCAN(eps = 1, min_samples = 4).fit(x)
# labels = dbscan.labels_
#
#
# plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
#
# #***************
# csv_file4 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/Aggregation.csv')
# DF = pd.DataFrame(csv_file4)
#
#
# x = DF[["X","Y"]].values
#
# dbscan = DBSCAN(eps = 0.5, min_samples = 4).fit(x)
# # dbscan = DBSCAN(eps = 0.5, min_samples = 4).fit_predict(x)
# labels = dbscan.labels_
#
#
# plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
