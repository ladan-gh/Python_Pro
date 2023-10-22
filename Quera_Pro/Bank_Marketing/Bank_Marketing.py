import pandas as pd
from flaml import AutoML
from sklearn.metrics import f1_score
import numpy as np

#-----------------------------------------------------
df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/bank-additional-full.csv', na_values='unknown')
# print(df.head())

#-----------------------------------------------------
target_variable = 'y'
df_train = df.sample(frac=.7, random_state=313)

model = AutoML(task='classification', time_budget=120, verbose=0)
model.fit(df_train.drop(target_variable, axis=1), df_train[target_variable])

df_test = df.drop(df_train.index)
y_pred = model.predict(df_test.drop(target_variable, axis=1))

f1score = f1_score(df_test[target_variable], y_pred, pos_label='yes')*100
# print(f'performance of model is {f1score} %')

#---------------------part 1---------------------------------
categorical_cols =  ['job', 'marital', 'education', 'month', 'day_of_week', 'poutcome', 'default', 'housing', 'loan', 'is_telephone_contact']

for col in categorical_cols :
    df.loc[:,col] = df[col].fillna(df[col].mode().values[0])

# print(df.head())

#---------------------part 2---------------------------------
continuous_cols = df[['age', 'duration', 'campaign', 'pdays', 'previous']]
binary_cols = df[['default', 'housing', 'loan', 'is_telephone_contact']]
nominal_cols = df[['job', 'marital']]
ordinal_cols = df[['education', 'month', 'day_of_week', 'poutcome']]

#-------------------part 3------------------------------------
from sklearn import preprocessing
import pandas as pd

binary_cols = pd.DataFrame(binary_cols)
le = preprocessing.LabelEncoder()


for i in binary_cols.columns:
    binary_cols[i] = le.fit_transform(binary_cols[i])

binary_df = binary_cols
print(binary_df.head())

#---------------------part 4----------------------------------
sample_num = len(nominal_cols.values)

column_list_01 = []
column_list_02 = []
couter = 0

for i in range(0, len(nominal_cols.columns)):
    if couter == 0 :
        for j in np.unique(nominal_cols.values[:, [i]]):
            column_list_01.append(nominal_cols.columns[i] + '_' + j)
        couter += 1
    else:
        for j in np.unique(nominal_cols.values[:, [i]]):
            column_list_02.append(nominal_cols.columns[i] + '_' + j)

#-----------------part 4(one-hot encoding)---------------------
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()

data_01 = pd.DataFrame(one_hot.fit_transform(nominal_cols.job.values.reshape(-1, 1)).toarray())
data_01 = np.array(data_01)
nominal_one_hot_01 = pd.DataFrame(data_01, columns = column_list_01)
# print(nominal_one_hot_01)

data_02 = pd.DataFrame(one_hot.fit_transform(nominal_cols.marital.values.reshape(-1, 1)).toarray())
data_02 = np.array(data_02)
nominal_one_hot_02 = pd.DataFrame(data_02, columns = column_list_02)
# print(nominal_one_hot_02)

#------------------part 4(concat one-hot dataframe)----
frames = [nominal_one_hot_01, nominal_one_hot_02]
nominal_df = pd.concat(frames, axis=1)
# print(nominal_df.head())

#-------------------part 5------------------------------
mapping_education = {'illiterate': 0,
                     'basic.4y' : 1,
                     'basic.6y' : 2,
                     'basic.9y' : 3,
                     'high.school' : 4,
                     'professional.course' : 5,
                     'university.degree' : 6}

mapping_day_of_week = {'mon': 0,
                 'tue' : 1,
                 'wed' : 2,
                 'thu' : 3,
                 'fri' : 4,
                 'sat' : 5,
                 'sun' : 6}

mapping_month       = {'feb': 0,
                       'jan' : 1,
                       'mar' : 2,
                       'apr' : 3,
                       'may' : 4,
                       'jun' : 5,
                       'jul' : 6,
                       'aug' : 7,
                       'sep' : 8,
                       'oct' : 9,
                       'nov' : 10,
                       'dec' : 11}


mapping_poutcome = {'failure' : -1,
                    'nonexistent' : 0,
                    'success' : 1}

ordinal_df = pd.DataFrame()
ordinal_df['education'] = ordinal_cols['education'].map(mapping_education)
ordinal_df['month'] = ordinal_cols['month'].map(mapping_month)
ordinal_df['day_of_week'] = ordinal_cols['day_of_week'].map(mapping_day_of_week)
ordinal_df['poutcome'] = ordinal_cols['poutcome'].map(mapping_poutcome)
# print(ordinal_df.head())

#------------------------------------------------------------
import zipfile
import joblib

joblib.dump(categorical_cols, "C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/categorical_cols")
joblib.dump(continuous_cols, "C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/continuous_cols")
joblib.dump(binary_cols, "C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/binary_cols")
joblib.dump(nominal_cols, "C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/nominal_cols")
joblib.dump(ordinal_cols, "C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/ordinal_cols")


binary_df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/binary_df.csv', index=False)
nominal_df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/nominal_df.csv', index=False)
ordinal_df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/ordinal_df.csv', index=False)
df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/df.csv', index=False)


def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Bank_Marketing/' + file_name, file_name, compress_type=compression)

file_names = ["categorical_cols", "continuous_cols", "binary_cols", "nominal_cols", "ordinal_cols", \
              "binary_df.csv", "nominal_df.csv", "ordinal_df.csv", "df.csv", "Bank_Marketing.py"]

compress(file_names)