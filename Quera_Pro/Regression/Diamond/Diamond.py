import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#-------------------------------------------------------
train = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/diamonds_train.csv')
# print(train.head())

test = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/diamonds_test.csv') # TO-DO
# print(test.head())
# print(train.info())

#string columns: cut, color, clarity
#----------------------------------------------------
# Do some preprocessing!
le = LabelEncoder()

train['cut'] = le.fit_transform(train['cut'])
train['color'] = le.fit_transform(train['color'])
train['clarity'] = le.fit_transform(train['clarity'])

test['cut'] = le.fit_transform(test['cut'])
test['color'] = le.fit_transform(test['color'])
test['clarity'] = le.fit_transform(test['clarity'])

# ---------Create Model----------------------
from sklearn.linear_model import LinearRegression

x_train = train[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y_train = train['price']
model = LinearRegression().fit(x_train, y_train)

#--------------------------------------------
from sklearn.metrics import r2_score

x_test = train[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y_test = train['price']
y_pred = model.predict(x_test)
r2_score(y_test, y_pred)
# print(r2_score(y_test, y_pred))

#------------------------------------------------
# predict test samples
submission = pd.DataFrame()
predicted = model.predict(test)
submission['price'] = predicted

# print(submission)

#-------------------------------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/' + file_name, file_name, compress_type=compression)

submission.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/submission.csv', index=False)
joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Diamond/model')
file_names = ['Diamond.py', 'submission.csv', 'model']
compress(file_names)