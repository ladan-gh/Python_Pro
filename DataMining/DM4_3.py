import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#================Q3
from sklearn.cluster import AgglomerativeClustering


csv_file1 = pd.read_csv('C:/Users/Lenovo/Desktop/spiral.csv')
DF = pd.DataFrame(csv_file1)

hc = AgglomerativeClustering(n_clusters = 5)
X = DF[["X","Y"]].values
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('Hierarchical Diagram(spiral)')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#***************
csv_file2 = pd.read_csv('C:/Users/Lenovo/Desktop/jain.csv')
DF = pd.DataFrame(csv_file2)

hc = AgglomerativeClustering(n_clusters = 5)
X = DF[["X","Y"]].values
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('Hierarchical Diagram(jain)')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#***************
csv_file3 = pd.read_csv('C:/Users/Lenovo/Desktop/flame.csv')
DF = pd.DataFrame(csv_file3)

hc = AgglomerativeClustering(n_clusters = 5)
X = DF[["X","Y"]].values
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('Hierarchical Diagram(flame)')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#***************
csv_file4 = pd.read_csv('C:/Users/Lenovo/Desktop/Aggregation.csv')
DF = pd.DataFrame(csv_file4)

hc = AgglomerativeClustering(n_clusters = 5)
X = DF[["X","Y"]].values
y_hc=hc.fit_predict(X)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.title('Hierarchical Diagram(Aggregation)')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()