import numpy as np
import pandas as pd
from khayyam import JalaliDatetime
from khayyam import JalaliDate
from datetime import timedelta
from flaml import AutoML
from sklearn.metrics import r2_score
import calendar
from datetime import date

#----------------------------------------
df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Traffic_Pro/traffic.csv', parse_dates=['DateTime'])
# print(df)

#----------------------------------------
# model = AutoML(task='regression', time_budget=60, verbose=0)
# train = df.loc[:48000]
# test = df.loc[48000:]
# model.fit(train.drop('Car', axis=1), train.Car)
# y_pred = model.predict(test.drop('Car', axis=1))
# r2score = round(r2_score(test.Car, y_pred),2) * 100
# print(f'performance of model is {r2score}%')

#--------------part 1---------------------------
date_ = []
for i in range(0, len(df)):
    miladi_date = df['DateTime'][i]
    shamsi_date = JalaliDatetime(miladi_date)
    date_.append(shamsi_date)

df['JalaliDateTime'] = date_
# print(df)
#--------------part 2---------------------------
# df['hour'] = None
sample_num = len(df)
df['hour'] = np.zeros(sample_num)
df['new_time'] = [d.time() for d in df['DateTime']]

#--------------------------
j = 0
for i in df['new_time']:

    if i >= df['new_time'][0] and i < df['new_time'][6]:
        df['hour'][j] = 0

    if i >= df['new_time'][6] and i < df['new_time'][12]:
        df['hour'][j] = 1

    if i >= df['new_time'][12] and i < df['new_time'][15]:
        df['hour'][j] = 2

    if i >= df['new_time'][15] and i < df['new_time'][18]:
        df['hour'][j] = 3

    if i >= df['new_time'][18] and i < df['new_time'][22]:
        df['hour'][j] = 4

    if i >= df['new_time'][22] and i > df['new_time'][0]:
        df['hour'][j] = 5


    j += 1

# print(df['hour'])
#----------part 3---------------------------------
df['IsHoliday'] = np.zeros(sample_num)
df['new_date'] = [d.date() for d in df['JalaliDateTime']]


for i in df['new_date']:
    source_date = JalaliDate(i)
    print(source_date.weekday())


k = 0
for i in df['new_date']:
    # x = calendar.day_name[i.weekday()]
    source_date = JalaliDate(i)
    x = source_date.weekday()

    if x == 6:
        df['IsHoliday'][k] = 1
    else:
        df['IsHoliday'][k] = 0

    k += 1

# print(df['IsHoliday'])
#----------part 4---------------------------------
df['IsCold'] = np.zeros(sample_num)


k = 0
for i in df['new_date']:
    source_date = JalaliDate(i)
    month = source_date.month

    if month > 6 and month <= 12:
        df['IsCold'][k] = 1

    else:
        df['IsCold'][k] = 0

    k += 1

# print(df['IsCold'])

#------------part 5--------------------
sample_num = len(df)

column_list_ = []

for j in np.unique(df['Junction'].values):
    column_list_.append('Junc' + '_' + str(j))

#----------------------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_ = pd.DataFrame(one_hot.fit_transform(df.Junction.values.reshape(-1, 1)).toarray())
data_ = np.array(data_)
nominal_one_hot_ = pd.DataFrame(data_, columns = column_list_)
# print(nominal_one_hot_)

frames = [df, nominal_one_hot_]
df = pd.concat(frames, axis=1)
# print(nominal_df.head())
#-----------------------------------
# import zipfile
# import joblib
#
# synthesized_cols = ['JalaliDateTime', 'hour', 'IsHoliday', 'IsCold', 'Junc_1', 'Junc_2', 'Junc_3', 'Junc_4']
#
# submision_df = df[synthesized_cols]
#
# submision_df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Traffic_Pro/df.csv', index=False)
#
#
# def compress(file_names):
#     print("File Paths:")
#     print(file_names)
#     compression = zipfile.ZIP_DEFLATED
#     with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Traffic_Pro/result.zip", mode="w") as zf:
#         for file_name in file_names:
#             zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Traffic_Pro/' + file_name, file_name, compress_type=compression)
#
#
# file_names = ["df.csv", "Traffic.py"]
# compress(file_names)