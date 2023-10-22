import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#-----------------------------------
train = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/nba_players_train.csv')
test = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/nba_players_test.csv')

#---------------------------
# do some preprocessing!
# String Columns: name
le = LabelEncoder()

train['name'] = le.fit_transform(train['name'])
test['name'] = le.fit_transform(test['name'])


#-----------------------------
# modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

x_train = train[['name', 'gp', 'min', 'pts', 'fgm', 'fga', 'fg', '3p_made', '3pa', '3p',
            'ftm', 'fta', 'ft', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov']]

y_train = train['target_5yrs']

model = LogisticRegression(penalty='l2', fit_intercept=True, class_weight='balanced', max_iter=20)
model.fit(x_train, y_train)

#-------------------------------------------
# predict test samples
submission = pd.DataFrame()
pre = model.predict(test)
submission['target_5yrs'] = pre

#--------------------------------
import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)

    compression = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile("C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/result.zip", mode="w") as zf:
        for file_name in file_names:

            zf.write('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/' + file_name, file_name, compress_type=compression)



joblib.dump(model, 'C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/model')
submission.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NBA/submission.csv', index=False)

file_names = ['model', 'submission.csv', 'NBA.py']
compress(file_names)