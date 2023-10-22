import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#------------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/train.csv')
test_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/test.csv')

x_train = train_data.iloc[:, 1:]
y_train = train_data['class']

#-------------------------
# do some preprocessing!
le = LabelEncoder()

for i in x_train.columns:
    x_train[i] = le.fit_transform(x_train[i])

for i in test_data.columns:
    test_data[i] = le.fit_transform(test_data[i])

#------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

#-----------------------------
# predict test samples
prediction = pd.DataFrame()
predicted = model.predict(test_data)
prediction['class'] = predicted

#-----------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/' + file_name, file_name, compress_type=compression)

joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/model')
prediction.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/Mushroom_Hunter/submission.csv', index=False)
file_names = ['Mushroom_Hunter.py', 'submission.csv', 'model']
compress(file_names)