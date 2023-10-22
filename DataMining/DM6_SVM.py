import pandas as pd

#======================SVM
csv_file = pd.read_csv('C:/Users/Lenovo/Desktop/bill_authentication.csv')
df = pd.DataFrame(csv_file)

#======================
import numpy as np
import matplotlib.pyplot as plt

X = df.iloc[:, 0:4].values #rows
y = df.iloc[:, 4].values #lable

#======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#=========================
from sklearn.svm import SVC

#svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel='rbf')

svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)

#==================================
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

MA = metrics.accuracy_score(y_test, y_pred)
print("MA is :")
print(MA)
print("#================")


CM = confusion_matrix(y_test, y_pred)
print("CM is :")
print(CM)
print("#================")

print("classification_report is :")
print(classification_report(y_test, y_pred))



