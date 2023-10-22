import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#================Q2
from sklearn.cluster import KMeans
#
# csv_file1 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/spiral.csv')
# DF = pd.DataFrame(csv_file1)
#
#
# X = DF[["X","Y"]]
# km = KMeans(n_clusters=3, n_init = 3)
# km.fit(X)
# y_kmeans = km.predict(X)
#
#
# sns.scatterplot(data=DF, x="X", y="Y", hue= y_kmeans, palette= "coolwarm")
# centers = km.cluster_centers_
# #Plot centers
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha = 0.6)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
#
#
# print(km.inertia_) #Min is better

# #****************
# csv_file2 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/jain.csv')
# DF = pd.DataFrame(csv_file2)
#
#
# X = DF[["X","Y"]]
# km = KMeans(n_clusters=3, n_init = 3)
# km.fit(X)
# y_kmeans = km.predict(X)
#
#
# sns.scatterplot(data=DF, x="X", y="Y", hue= y_kmeans, palette= "coolwarm")
# centers = km.cluster_centers_
#
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha = 0.6)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

#****************
# csv_file3 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/flame.csv')
# DF = pd.DataFrame(csv_file3)
#
#
# X = DF[["X","Y"]]
# km = KMeans(n_clusters=3, n_init = 3)
# km.fit(X)
# y_kmeans = km.predict(X)
#
#
# sns.scatterplot(data=DF, x="X", y="Y", hue= y_kmeans, palette= "coolwarm")
# centers = km.cluster_centers_
#
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha = 0.6)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

#****************
csv_file4 = pd.read_csv('C:/Users/Ladan_Gh/Desktop/clustering/Aggregation.csv')
DF = pd.DataFrame(csv_file4)


X = DF[["X","Y"]]
km = KMeans(n_clusters=7, n_init = 7, init = "random", random_state = 42, max_iter=200)
km.fit(X)
y_kmeans = km.predict(X)


sns.scatterplot(data=DF, x="X", y="Y", hue= y_kmeans, palette= "coolwarm")
centers = km.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha = 0.6)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()