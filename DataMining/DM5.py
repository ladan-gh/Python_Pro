import pandas as pd

#=====================Q1
csv_file = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/DataMining/penguins.csv')
df = pd.DataFrame(csv_file)

#print(df.count())
#print(df.dropna())
#print(df.count())

df = df.dropna()

#=====================Q2 to Q6
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt


#Importing the dataset

X = df.iloc[:, 1:].values #rows
y = df.iloc[:, 0].values #lable

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

#The above script splits the dataset into 80% train data and 20% test data. This means that out of total 150 records,
# the training set will contain 120 records and the test set contains 30 of those records.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)


#==================================Encoding--> test data and train data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[5] = le.fit_transform(X_train[5])
X_train[0] = le.fit_transform(X_train[0])

X_test[5] = le.fit_transform(X_test[5])
X_test[0] = le.fit_transform(X_test[0])


#==================================
#Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#==================================Create K-NN model
#Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#==================================
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))


error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()