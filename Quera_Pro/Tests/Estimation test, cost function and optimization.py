# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# #---------------------------------------
# df = pd.read_csv('C:/Users/Ladan_Gh/Desktop/data.csv')
# #
# # df['x1'] = np.ones(4)
# # x = df[['x', 'x0', 'x1']]
# # y = df[['y']]
# #
# # x1 = x['x']
# # x1 = np.matrix(x1)
# # x2 = x['x0']
# # x2 = np.matrix(x2)
# #
# #
# # x_01 = np.matrix(x)
# # y_01 = np.matrix(y)
# #
# #
# # x_transpose = np.transpose(x_01)
# # x_ = x_transpose * x_01
# # x_inverse = np.linalg.inv(x_)
# #
# #
# # w0 = 1
# # w1 = 0.5
# # w2 = 0.5
# #
# # y_pred = w0 + (w1 * x1) + (w2 * x2)
# #
# # j_theta = (1/(2 * len(x))) * np.sum((y_pred - y_01) ** 2) # MSE
# #
# # print(j_theta)
# #
# #--------------------------------------------------
# def Gradient_Descent(x ,y):
#
#     x = np.matrix(x)
#     y = np.matrix(y)
#
#     theta_new = [[0], [0]]
#     # theta_new = 0
#     theta_new = np.matrix(theta_new)
#
#     theta_0 = 0
#     theta_1 = 0
#     # num of data point
#     n = len(x)
#
#     # initialize the learning rate
#     l = 0.1
#
#     for i in range(0, 1):
#
#         y_pred = theta_0 + (theta_1 * x)
#
#         theta_old = theta_new
#         theta_new = theta_old - (np.dot(l, np.sum(np.dot((y - y_pred), x))))
#
#         cost = (1 / 2) * np.sum((y - y_pred) ** 2)  # MSE
#
#         if theta_new == theta_old:
#             break
#
#
# y = df['y']
# x = df['x']
#
# Gradient_Descent(x, y)

#--------------------------------------------
# h = w0 + (w1 * x1) + (w2 * x2)

hyp = []
x1 = [2, 1, 3, 0]
x2 = [4, 0, 4, 2]
y_true = [4, 1, 5, 1]

w0 = [0, 0, 1]
w1 = [1, 0.5, 0.5]
w2 = [1, 0.5, 0.5]

for i in range(0, len(x1)):
    h = w0[0] + (w1[0] * x1[i]) + (w2[0] * x2[i])
    y = h - y_true[i]
    hyp.append(y**2)

sum_ = sum(hyp)
print(sum_)











