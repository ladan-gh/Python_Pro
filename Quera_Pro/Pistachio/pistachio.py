import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math

#-----------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/train.csv')
test_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/test.csv')

# print(train_data)
#-----------------------------
# String columns: Class
le = LabelEncoder()
train_data['Class'] = le.fit_transform(train_data['Class'])

#---------------------------
# import seaborn as sns
# import matplotlib.pyplot as plt

# outlier = ['MINOR_AXIS', 'ECCENTRICITY', 'SOLIDITY', 'EXTENT', 'ASPECT_RATIO', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_3', 'SHAPEFACTOR_4']
# not_outlier = list()

# for i in train_data.columns:
#
#     sns.boxplot(x=train_data[i])
#     plt.suptitle(i)
#     plt.show()

# for i in test_data.columns:
#         if i in outlier:
#             continue
#         else:
#             not_outlier.append(i)

#-----------------------------------
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
#

# x_train = train_data.iloc[:, :-1]
# train_data[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'EQDIASQ', 'CONVEX_AREA', 'ROUNDNESS', 'SHAPEFACTOR_2']] = sc.fit_transform(x_train[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'EQDIASQ', 'CONVEX_AREA', 'ROUNDNESS', 'SHAPEFACTOR_2']])

#---------------------------
# for i in train_data.columns:
#     print('min '+ i + ' is:', min(train_data[i]))
#     print('max '+ i + ' is:', max(train_data[i]))
#     print('*************************')

#, 'PERIMETER', 'EQDIASQ', 'MINOR_AXIS', 'MAJOR_AXIS'
#
# train_data = train_data.drop(columns=['AREA', 'CONVEX_AREA','PERIMETER', 'EQDIASQ', 'MINOR_AXIS', 'MAJOR_AXIS'])
# test_data = test_data.drop(columns=['AREA', 'CONVEX_AREA','PERIMETER', 'EQDIASQ', 'MINOR_AXIS', 'MAJOR_AXIS'])

#------------------------------
from sklearn.neighbors import KNeighborsClassifier

x_train_01 = train_data.iloc[:, :16]
y_train_01 = train_data.iloc[:, 16]


print(x_train_01)
# print(y_train_01)

model = KNeighborsClassifier(n_neighbors = 42)
model.fit(x_train_01, y_train_01)

# -------------------------------
# predict test samples
# prediction = pd.DataFrame()
# predicted = model.predict(test_data)
# prediction['Class'] = predicted
# prediction['Class'] = le.inverse_transform(prediction['Class'])
#
# # ------------------------------
# import zipfile
# import joblib
#
# def compress(file_names):
#     print("File Paths:")
#     print(file_names)
#     compression = zipfile.ZIP_DEFLATED
#     with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/result.zip", mode="w") as zf:
#         for file_name in file_names:
#             zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/' + file_name, file_name, compress_type=compression)
#
#
# joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/model')
# prediction.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/submission.csv', index=False)
# file_names = ['pistachio.py', 'submission.csv', 'model']
# compress(file_names)