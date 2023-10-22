import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

#--------------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/fraud_train.csv')
test_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/fraud_test.csv')

x_train = train_data.iloc[:, :30]
y_train = train_data['Class']

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)

#--------------------------
# do some preprocessing!
# calculating quantiles to detect outliers
Q1, Q2, Q3 = 0.25, 0.5, 0.75

for i in x_train.columns:

    Q1 = x_train[i].quantile(0.25)
    Q3 = x_train[i].quantile(0.75)
    fence = 1.5 * (Q3 - Q1)

    plot_df = x_train[(x_train[i] >= Q1 - fence) & (x_train[i] <= Q3 + fence)] # Data don't need removal
    outlier_ = x_train[(x_train[i] < Q1 - fence) | (x_train[i] > Q3 + fence)]
    outlier_ = outlier_.sort_values(by=i, ascending=True)[i].tolist()

    for k in range(0, len(outlier_)):
        for j in range(len(x_train[i])):
            if outlier_[k] == x_train[i][j]:
                newdf = x_train.drop(columns=i)

#-------------Model-----------------
model = SVC(kernel='linear')
model.fit(x_train, y_train)
# y_pred = model.predict(test_data)

#------------------------------------
submission = pd.DataFrame()
predicted = model.predict(test_data)
submission['Class'] = predicted
print(submission)

#---------------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/' + file_name, file_name, compress_type=compression)


joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/model')
submission.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/fraud_detection/submission.csv', index=False)

file_names = ['model', 'submission.csv', 'fraud_detection.py']
compress(file_names)