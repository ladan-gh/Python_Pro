import pandas as pd

#========================
#1)
csv_file = pd.read_csv('C:/Users/Lenovo/Desktop/Ladan.csv')
print(csv_file)
print('#=============')
#=============================
#2)
df = pd.DataFrame(csv_file)
sum = 0
length = len(df['Grades'])

for i in range(0,length):
    unit = df['Units'].values[i]
    x = unit * df['Grades'].values[i]
    sum += x

number = len(df['Units'])
count = 0
for i  in range(0,number):
    count += df['Units'].values[i]

avg = sum/count
print('my average is :', avg)
print('#=============')
#============================
#3)
counter = 0
length = len(df['Grades'])

for i in range(0,length):
    x = df['Grades'].values[i]
    if x > 12 :
        counter += 1

print('Number of scores above 12 :', counter)




"""print(df.isna().sum())
print("#++++++++++")

#count total missing values in a dataframe
print(df.dropna().isna().sum())"""