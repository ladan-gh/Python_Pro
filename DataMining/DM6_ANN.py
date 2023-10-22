import pandas as pd

#============================
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

#=====================
csv_file = pd.read_csv('C:/Users/Lenovo/Desktop/bill_authentication.csv')
df = pd.DataFrame(csv_file)

#=====================
import numpy as np
import matplotlib.pyplot as plt

X = df.iloc[:, 0:4].values #rows
y = df.iloc[:, 4].values #lable

#======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print("x_test type:")
print(type(X_test))
print("x_train type:")
print(type(X_train))
print("y_test type:")
print(type(y_test))
print("y_train type:")
print(type(y_train))


#======================
"""from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier(hidden_layer_sizes=(20), max_iter = 300,activation = 'relu',solver = 'adam')

MLP.fit(X_train,y_train)
y_pred = MLP.predict(X_test)

#=========================
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

MA = accuracy_score(y_test, y_pred)
print("MA is :")
print(MA)
print("#================")

CM = confusion_matrix(y_test, y_pred)
print("CM is :")
print(CM)
print("#================")

print("classification_report is :")
print(classification_report(y_test, y_pred))


fig = plot_confusion_matrix(MLP, X_test, y_test, display_labels=MLP.classes_)
fig.figure_.suptitle("Confusion Matrix for bill_authentication Dataset")
plt.show()

plt.plot(MLP.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
#===========================
Acc = []
Neural = []

for i in range(2, 21):
    MLP = MLPClassifier(hidden_layer_sizes=(i) ,max_iter = 300 ,activation = 'relu' ,solver = 'adam')
    MLP.fit(X_train, y_train)
    y_pred = MLP.predict(X_test)
    Acc.append(accuracy_score(y_test, y_pred))
    Neural.append(i)


plt.figure(figsize=(12, 6))
plt.plot(Neural, Acc, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Accuracy_NeuronsCount')
plt.xlabel('NeuronsCount')
plt.ylabel('Accuracy')
plt.show()
"""
