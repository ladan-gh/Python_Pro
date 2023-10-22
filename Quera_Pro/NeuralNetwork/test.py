import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from inspect import getsource
import random

#--------------------
train_data = pd.read_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NeuralNetwork/iris.csv')
# print(train_data['Species'].unique())

train_data = train_data.drop(columns=['Id'])

#-------------------------
idx_one = np.where(train_data['Species'] == 'Iris-setosa')[0]
idx_two = np.where(train_data['Species'] == 'Iris-versicolor')[0]

frames = [train_data.iloc[idx_one], train_data.iloc[idx_two]]
df1 = pd.concat(frames)

#===================
idx_one = np.where(train_data['Species'] == 'Iris-setosa')[0]
idx_two = np.where(train_data['Species'] == 'Iris-virginica')[0]

frames = [train_data.iloc[idx_one], train_data.iloc[idx_two]]
df2 = pd.concat(frames)

#===================
idx_one = np.where(train_data['Species'] == 'Iris-versicolor')[0]
idx_two = np.where(train_data['Species'] == 'Iris-virginica')[0]

frames = [train_data.iloc[idx_one], train_data.iloc[idx_two]]
df3 = pd.concat(frames)

#--------------------------------------------
# df1.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NeuralNetwork/df1.csv', index=False)
# df2.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NeuralNetwork/df2.csv', index=False)
# df3.to_csv('C:/Users/Ladan_Gh/PycharmProjects/Quera_Pro/NeuralNetwork/df3.csv', index=False)

#--------------------------------
# To-Do
x_train_01 = df1.iloc[:, 1:4]
y_train_01 = df1['Species']

mapping_Species_01 = {'Iris-setosa' : 1,
                    'Iris-versicolor' : -1}

y_train_01 = y_train_01.map(mapping_Species_01)
# print(y_train_01)
#============================
x_train_02 = df2.iloc[:, 1:4]
y_train_02 = df2['Species']

mapping_Species_02 = {'Iris-setosa' : 1,
                    'Iris-virginica' : -1}

y_train_02 = y_train_02.map(mapping_Species_02)


# print(x_train_02.loc[1])
# print(y_train_02)
#============================
x_train_03 = df3.iloc[:, 1:4]
y_train_03 = df3['Species']

mapping_Species_03 = {'Iris-versicolor' : 1,
                    'Iris-virginica' : -1}

y_train_03 = y_train_03.map(mapping_Species_03)
# print(y_train_03)

#--------------------------------
class Perceptron:

    def __init__(self):
        self.weights = None
        # self.epochs = epochs
        # self.bias = None
        # self.learning_rate = learning_rate


    def weighting(self, input):
        weighted_input = np.dot(input, self.weights)
        return weighted_input


    def activation(self, weighted_input):
        y_pred = list()

        for i in range(0, len(weighted_input)):
            if weighted_input[i] >= 0:
                y_pred.append(1)
                # return 1
            else:
                y_pred.append(-1)
                # return -1
        return y_pred


    def predict(self, inputs):
        inputs['bias'] = 1

        weighted_input = self.weighting(inputs)
        label = [self.activation(weighted_input)]

        return label


    def fit(self, inputs, outputs, learning_rate, epochs):
        from sklearn.metrics import accuracy_score

        inputs['bias'] = 1

        n_features = inputs.shape[1]
        self.weights = np.zeros((n_features))

        for i in range(0, n_features):
            self.weights[i] = random.randint(0, 1)


        for epoch in range(epochs):
            for i in range(len(inputs)):

                z = np.dot(inputs, self.weights) #+ inputs['bias'][i]
                y_pred = self.activation(z)

                # for i in range(0, len(z)):
                #     y_pred.append(self.activation(z[i]))


                # Updating weights and bias
                self.weights = self.weights + learning_rate * (outputs[i] - y_pred[i]) * inputs.loc[i]

            # accuracy_score(y_true, y_pred)
        pass

#---------------------------------
# perceptron = Perceptron()
# perceptron.fit(x_train_01, y_train_01, 0.1, 10)

perceptron = Perceptron()
perceptron.fit(x_train_02, y_train_02, 0.1, 10)

# perceptron = Perceptron()
# perceptron.fit(x_train_03, y_train_03, 0.1, 10)