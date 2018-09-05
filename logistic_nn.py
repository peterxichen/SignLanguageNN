# logistic regression implemented as one layer neural network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import data
x_data = np.load('./X.npy')
y_data = np.load('./Y.npy')

# display an image
# plt.imshow(x[0].reshape(64,64))
# plt.show

# train for only 0's (204-409) and 1's (822-1027)
x = np.concatenate((x_data[204:409], x_data[822:1027] ), axis=0)
y = np.concatenate((np.zeros(205), np.ones(205)),axis=0).reshape(410,1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

# flatten image arrays and transpose
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).T
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).T
y_train = y_train.T
y_test = y_test.T

# build out logistic regression
#z = (weight)x+b -> pass through sigmoid function
# Parameters: weight (coefficients of pixels) and bias (intercept)

# initialize parameters where dim = number of pixels
def initWeightBias(dim):
    w = np.full((dim,1),0.01)
    b = 0
    return w, b

# outputs probability using sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# forward propogation: calculate and sum loss(error) function
def fwdProp(w,b,x_train, y_train):
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    # log loss function
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    return cost

# Gradient Descent: update parameters to minmiize cost
def prop(w, b, x_train, y_train):
    # forward
    z = np.dot(w.T, x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward
    der_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    der_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"der_weight": der_weight,
                 "der_bias": der_bias}
    return cost,gradients

# learning the parameters
def update(w, b, x_train, y_train, learning_rate, iterations):
    cost1 = []
    cost2 = []
    index = []
    # update parameters
    for i in range(iterations):
        cost, gradients = prop(w,b,x_train,y_train)
        cost1.append(cost)
        # update
        w = w-learning_rate*gradients["der_weight"]
        b = b-learning_rate*gradients["der_bias"]
        if (i%10==0):
            cost2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost2) # plot descent
    plt.xlabel("Itrations")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost1

# make prediction
def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_pred = np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if (z[0,i]<=0.5):
            y_pred[0,i]=0
        else:
            y_pred[0,i]=1
    return y_pred

# perform logistic regression (using a NN)
def logreg(x_train, y_train, x_test, y_test, learn_rate, iterations):
    dim = x_train.shape[0]
    w, b = initWeightBias(dim)
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learn_rate, iterations)    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

# run
logreg(x_train, y_train, x_test, y_test,learn_rate = 0.01, iterations = 100)

#run using sklearn
logreg_sk = LogisticRegression(max_iter= 100)
print("test accuracy: {} ".format(logreg_sk.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg_sk.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
