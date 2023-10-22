import numpy as np
import pandas as pd
from flaml import AutoML
from matplotlib import pyplot as plt
import seaborn as sns

#----------------------------------------
# read csv file
df = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/GPS_Pro/travel_times.csv')
# print(df.info())

#-----------------------------------------
# drop comments column
# df = df.drop('Comments', axis=1)
# df.info()

#-----------------------------------------
# fill NaN values in the AvgMovingSpeed and FuelEconomy

avg_01 = df['AvgMovingSpeed'].mean().round(2)

df['AvgMovingSpeed'].fillna(avg_01, inplace=True)



avg_02 = df['FuelEconomy'].mean().round(2)
df['FuelEconomy'].fillna(avg_02, inplace=True)
# print(df)

#-----------------------------------------
automl_reg = AutoML()
x_train, x_test = df.dropna(subset=['TotalTime']), df[df.TotalTime.isna()]
y_train = df[~df.TotalTime.isna()].TotalTime
automl_reg.fit(x_train, y_train, task="regression", verbose=False)
y_pred = automl_reg.predict(x_test)


df.loc[df.TotalTime.isna(), 'TotalTime'] = y_pred
# df.info()

#-------------------------------------------------
automl_clf = AutoML()
x_train, x_test = df.dropna(subset=['Toll']), df[df.Toll.isna()]
y_train = df[~df.Toll.isna()].Toll
automl_clf.fit(x_train, y_train, task="classification", verbose=False)
y_pred = automl_clf.predict(x_test)

df.loc[df.Toll.isna(), 'Toll'] = y_pred
# df.info()

#--------------------------------------------------
fig, ax = plt.subplots(4, 2, figsize=(20, 15))
sns.boxplot(x=df.Distance, ax = ax[0,0])
sns.boxplot(x=df.MaxSpeed, ax = ax[0,1])
sns.boxplot(x=df.AvgSpeed, ax = ax[1,0])
sns.boxplot(x=df.AvgMovingSpeed, ax = ax[1,1])
sns.boxplot(x=df.FuelEconomy, ax = ax[2,0])
sns.boxplot(x=df.TotalTime, ax = ax[2,1])
sns.boxplot(x=df.MovingTime, ax = ax[3,0])
plt.show()

#---------------------------------------------------
# calculating quantiles to detect outliers
Q1, Q2, Q3 = df.MaxSpeed.quantile([0.25, 0.5, 0.75])

Q1 = df['MaxSpeed'].quantile(0.25)
Q3 = df['MaxSpeed'].quantile(0.75)
fence = 1.5 * (Q3 - Q1)

plot_df = df[(df['MaxSpeed'] >= Q1 - fence) & (df['MaxSpeed'] <= Q3 + fence)] # Data don't need removal
outlier_speed = df[(df['MaxSpeed'] < Q1 - fence) | (df['MaxSpeed'] > Q3 + fence)]
outlier_speed = outlier_speed.sort_values(by='MaxSpeed', ascending=True)['MaxSpeed'].tolist()


#-------------------------------------------------------------
mean_ = plot_df['MaxSpeed'].mean().round(2)

for i in range(0, len(outlier_speed)):
    for j in range(len(df['MaxSpeed'])):
        if outlier_speed[i] == df['MaxSpeed'][j]:
            df.loc[j, 'MaxSpeed'] = mean_

# print('finish:)')


#---------------------------Show BoxPlot Without Outliers---------------------
fig, ax = plt.subplots(4, 2, figsize=(20, 15))
# sns.boxplot(x=df.Distance, ax = ax[0,0])
sns.boxplot(x=df.MaxSpeed, ax = ax[0,1])
# sns.boxplot(x=df.AvgSpeed, ax = ax[1,0])
# sns.boxplot(x=df.AvgMovingSpeed, ax = ax[1,1])
# sns.boxplot(x=df.FuelEconomy, ax = ax[2,0])
# sns.boxplot(x=df.TotalTime, ax = ax[2,1])
# sns.boxplot(x=df.MovingTime, ax = ax[3,0])
plt.show()

#------------------------------------------------------------
import zlib
import zipfile

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/GPS_Pro/result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/GPS_Pro/' + file_name, file_name, compress_type=compression)

np.savez("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/GPS_Pro/answers.npz", outlier_speed= outlier_speed)
df.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/GPS_Pro/preprocessed_df.csv',index = True)

file_names = ["answers.npz", "preprocessed_df.csv", "GPS_Pro.py"]
compress(file_names)