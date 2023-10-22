import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#---------------------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/train.csv')
test_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/test.csv')

# print(train_data.info())
# print(test_data.info())
#---------------------------------
# do some preprocessing!
# String columns: Country, Status
le = LabelEncoder()

train_data['Country'] = le.fit_transform(train_data['Country'])
train_data['Status'] = le.fit_transform(train_data['Status'])

test_data['Country'] = le.fit_transform(test_data['Country'])
test_data['Status'] = le.fit_transform(test_data['Status'])

#=================
# Null Columns(tarin data): Population, Hepatitis B, Polio, Diphtheria, Total expenditure, GDP, BMI, thinness  1-19 years, Alcohol, Schooling
# Null Columns(test data): Hepatitis B, Total expenditure, GDP, Alcohol

avg_01 = train_data['Population'].mean().round(2)
train_data['Population'].fillna(avg_01, inplace=True)

avg_02 = train_data['Hepatitis B'].mean().round(2)
train_data['Hepatitis B'].fillna(avg_02, inplace=True)

avg_03 = train_data['Polio'].mean().round(2)
train_data['Polio'].fillna(avg_03, inplace=True)

avg_04 = train_data['Diphtheria'].mean().round(2)
train_data['Diphtheria'].fillna(avg_04, inplace=True)

avg_05 = train_data['Total expenditure'].mean().round(2)
train_data['Total expenditure'].fillna(avg_05, inplace=True)

avg_06 = train_data['GDP'].mean().round(2)
train_data['GDP'].fillna(avg_06, inplace=True)

avg_07 = train_data['BMI'].mean().round(2)
train_data['BMI'].fillna(avg_07, inplace=True)

avg_08 = train_data['thinness  1-19 years'].mean().round(2)
train_data['thinness  1-19 years'].fillna(avg_08, inplace=True)

avg_09 = train_data['Alcohol'].mean().round(2)
train_data['Alcohol'].fillna(avg_09, inplace=True)

avg_10 = train_data['Schooling'].mean().round(2)
train_data['Schooling'].fillna(avg_10, inplace=True)

#===========================
avg_01 = test_data['Total expenditure'].mean().round(2)
test_data['Total expenditure'].fillna(avg_01, inplace=True)

avg_02 = test_data['Hepatitis B'].mean().round(2)
test_data['Hepatitis B'].fillna(avg_02, inplace=True)

avg_03 = test_data['GDP'].mean().round(2)
test_data['GDP'].fillna(avg_03, inplace=True)

avg_04 = test_data['Alcohol'].mean().round(2)
test_data['Alcohol'].fillna(avg_04, inplace=True)

#-------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x_train = train_data.iloc[:, 0:17]
y_train = train_data.iloc[:, 17]
x_test = test_data.iloc[:, :]

x_train = x_train.transpose()
x_test = x_test.transpose()


poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

model = LinearRegression()
model.fit(x_train_trans, y_train)
y_pred = model.predict(x_test_trans)
print(r2_score(y_test, y_pred))

#---------------------------------
submission = pd.DataFrame()
predicted = model.predict(x_test)
submission['Life expectancy'] = predicted

#-------------------------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Pistachio/' + file_name, file_name, compress_type=compression)


joblib.dump(poly_transformer, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/poly_transformer')
joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/model')
submission.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Regression/Life_Expectancy/submission.csv', index=False)

file_names = ['poly_transformer', 'model', 'submission.csv', 'Life_Expectancy.py']
compress(file_names)