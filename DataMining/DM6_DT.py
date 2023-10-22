import pandas as pd
import seaborn as sns

#======================DT
csv_file = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/ID3_pro/PlayTennis.csv')
df = pd.DataFrame(csv_file)

# df = df.dropna()
# feature_cols = df.columns.values

#======================
import numpy as np
import matplotlib.pyplot as plt

x = df.iloc[:, :4].values #rows
y = df.iloc[:, 4].values #lable

#======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#==================================Encoding--> test data and train data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[0] = le.fit_transform(X_train[0])
X_train[5] = le.fit_transform(X_train[5])


X_test[0] = le.fit_transform(X_test[0])
X_test[1] = le.fit_transform(X_test[1])

#==================================
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dtree = DecisionTreeClassifier()

#Train Decision Tree Classifer
dtree.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dtree.predict(X_test)

#==================================
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

MA = metrics.accuracy_score(y_test, y_pred)
print("MA is :")
print(MA)

#-----------plot confusion matrix--------------
CM = confusion_matrix(y_test, y_pred)
DetaFrame_cm = pd.DataFrame(CM, range(2), range(2))
sns.heatmap(DetaFrame_cm, annot=True)
plt.show()


print("classification_report is :")
print(classification_report(y_test, y_pred))

#==============plot DT==========================
dtree = DecisionTreeClassifier(criterion='gini', max_depth=4)
dtree.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(17, 8))
tree.plot_tree(dtree, fontsize=10)
plt.show()
