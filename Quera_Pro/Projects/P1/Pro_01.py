import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from khayyam import JalaliDatetime
from khayyam import JalaliDate
from datetime import timedelta

#---------------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Projects/P1/train_data.csv')
test_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Projects/P1/test_data.csv')

#**********************

x_train = train_data[['Created', 'CancelTime', 'DepartureTime', 'BillID', 'TicketID',
       'ReserveStatus', 'UserID', 'Male', 'Price', 'CouponDiscount', 'From',
       'To', 'Domestic', 'VehicleType', 'VehicleClass',
       'Vehicle', 'Cancel', 'HashPassportNumber_p', 'HashEmail', 'BuyerMobile',
       'NationalCode']]

y_train = train_data['TripReason']



print(x_train[['Domestic', 'VehicleClass',
       'Vehicle', 'Cancel','BuyerMobile']])
#----------------------------------
from datetime import datetime
date_01 = []
for i in range(0, len(x_train)):
    datetime_object = datetime.strptime(x_train['Created'][i], '%Y-%m-%d %H:%M:%S.%f')
    shamsi_date = JalaliDatetime(datetime_object)
    date_01.append(shamsi_date)

x_train['Created'] = date_01

#====================
from datetime import datetime
date_02 = []
for i in range(0, len(test_data)):
    datetime_object = datetime.strptime(test_data['Created'][i], '%Y-%m-%d %H:%M:%S.%f')
    shamsi_date = JalaliDatetime(datetime_object)
    date_02.append(shamsi_date)

test_data['Created'] = date_02

#----------------------------
# do some preprocessing
# Null value(train): CancelTime, UserID, VehicleType, VehicleClass, HashPassportNumber_p, HashEmail
# Null value(test): HashPassportNumber_p, HashEmail, VehicleClass, VehicleType, UserID, CancelTime

#main
x_train['VehicleType'].fillna(x_train['VehicleType'].mode(), inplace=True)
x_train['CancelTime'].fillna(x_train['CancelTime'].mode(), inplace=True)
x_train['VehicleClass'].fillna(x_train['VehicleClass'].mode(), inplace=True)
#main

x_train = x_train.drop('UserID', axis=1)
x_train = x_train.drop('HashPassportNumber_p', axis=1)
x_train = x_train.drop('HashEmail', axis=1)

#*******************************
#main
test_data['VehicleType'].fillna(test_data['VehicleType'].mode(), inplace=True)
test_data['CancelTime'].fillna(test_data['CancelTime'].mode(), inplace=True)
test_data['VehicleClass'].fillna(test_data['VehicleClass'].mode(), inplace=True)
#main

test_data = test_data.drop('UserID', axis=1)
test_data = test_data.drop('HashPassportNumber_p', axis=1)
test_data = test_data.drop('HashEmail', axis=1)

#------------TRAIN-----------------
column_list_ = []

for j in x_train['VehicleType'].unique().tolist():
    column_list_.append(j)

# print(column_list_)
#----------------------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_ = pd.DataFrame(one_hot.fit_transform(x_train.VehicleType.values.reshape(-1, 1)).toarray())
data_ = np.array(data_)
nominal_one_hot_ = pd.DataFrame(data_, columns = column_list_)
# print(nominal_one_hot_)

frames = [x_train, nominal_one_hot_]
df = pd.concat(frames, axis=1)

#-----------------------------
column_list_ = []

for j in df['Vehicle'].unique().tolist():
    column_list_.append(j)

# print(column_list_)
#----------------------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_ = pd.DataFrame(one_hot.fit_transform(df.Vehicle.values.reshape(-1, 1)).toarray())
data_ = np.array(data_)
nominal_one_hot_ = pd.DataFrame(data_, columns = column_list_)
# print(nominal_one_hot_)

frames = [df, nominal_one_hot_]
df = pd.concat(frames, axis=1)

#------------TEST-------------------
column_list_ = []

for j in test_data['VehicleType'].unique().tolist():
    column_list_.append(j)

# print(column_list_)
#----------------------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_ = pd.DataFrame(one_hot.fit_transform(test_data.VehicleType.values.reshape(-1, 1)).toarray())
data_ = np.array(data_)
nominal_one_hot_ = pd.DataFrame(data_, columns = column_list_)
# print(nominal_one_hot_)

frames = [test_data, nominal_one_hot_]
df = pd.concat(frames, axis=1)

#-----------------------------
column_list_ = []

for j in df['Vehicle'].unique().tolist():
    column_list_.append(j)

# print(column_list_)
#----------------------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_ = pd.DataFrame(one_hot.fit_transform(df.Vehicle.values.reshape(-1, 1)).toarray())
data_ = np.array(data_)
nominal_one_hot_ = pd.DataFrame(data_, columns = column_list_)
# print(nominal_one_hot_)

frames = [df, nominal_one_hot_]
df = pd.concat(frames, axis=1)
#===============================
# le = LabelEncoder()
#
# for i in df.columns:
#     df[i] = le.fit_transform(df[i])
#
# for i in test_data.columns:
#     test_data[i] = le.fit_transform(test_data[i])

# -----------------------------------
# modeling
model = RandomForestClassifier(n_estimators=200, max_depth=10)
model.fit(x_train, y_train)

# ---------------------------------
# predict test samples
submission = pd.DataFrame()
predicted = model.predict(test_data)
submission['TripReason'] = predicted

# ----------------------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Projects/P1/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Projects/P1/' + file_name, file_name, compress_type=compression)


submission.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Projects/P1/submission.csv', index=False)
file_names = ['Pro_01.py', 'submission.csv']
compress(file_names)