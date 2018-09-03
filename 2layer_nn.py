import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # neural network library
from keras.layers import Dense # layers library

# 2 layer neural network

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

# initializes parametrs + layers
def init(x_train, y_train):
    params = {"weight1": np.random.randn(3,x_train.shape[0])*0.1,
              "bias1": np.zeros((3,1)),
              "weight2": np.random.randn(y_train.shape[0],3)*0.1,
              "bias2": np.zeros((y_train.shape[0],1))}
    return params

# outputs probability using sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# forward propagation
def fwdProp(x_train, params):
    z1 = np.dot(params["weight1"],x_train)+params["bias1"]
    a1 = np.tanh(z1)
    z2 = np.dot(params["weight2"],a1)+params["bias2"]
    a2 = sigmoid(z2)
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache

# compute loss/cost function
def getCost(a2, y, params):
    cost = -np.sum(np.multiply(np.log(a2),y))/y.shape[1]
    return cost

# backwards propagation
def bckProp(params, cache, x, y):
    dz2 = cache["a2"]-y
    dw2 = np.dot(dz2, cache["a1"].T)/x.shape[1]
    db2 = np.sum(dz2, axis=1, keepdims=True)/x.shape[1]
    dz1 = np.dot(params["weight2"].T, dz2)*(1-np.power(cache["a1"],2))
    dw1 = np.dot(dz1,x.T)/x.shape[1]
    db1 = np.sum(dz1, axis=1, keepdims=True)/x.shape[1]
    grads = {"dweight1": dw1, "dbias1":db1, "dweight2": dw2, "dbias2": db2}
    return grads

# update parameters
def update(params, grads, rate=0.01):
    params = {
        "weight1": params["weight1"]-rate*grads["dweight1"],
        "bias1": params["bias1"]-rate*grads["dbias1"],
        "weight2": params["weight2"]-rate*grads["dweight2"],
        "bias2": params["bias2"]-rate*grads["dbias2"]
        }
    return params

# prediction (for classification)
def predict(params, x_test):
    a2, cache = fwdProp(x_test, params)
    y_pred = np.zeros((1, x_test.shape[1]))
    for i in range(a2.shape[1]):
        if a2[0,i]<=0.5:
            y_pred[0,i]=0
        else:
            y_pred[0,i]=1
    return y_pred

# create and run two layer neural network
def runNN(x_train, y_train, x_test, y_test, iters):
    costlist = []
    indexlist = []
    params = init(x_train, y_train)

    for i in range(0, iters):
        a2, cache = fwdProp(x_train, params)
        cost = getCost(a2, y_train, params)
        grads = bckProp(params, cache, x_train, y_train)
        params = update(params, grads)
        if (i%100)==0:
            costlist.append(cost)
            indexlist.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
    plt.plot(indexlist, costlist)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
        
    y_pred_test = predict(params, x_test)
    y_pred_train = predict(params, x_train)

    # print errors# Print train/test Errors
    print("Train accuracy: {} %".format(100-np.mean(np.abs(y_pred_train-y_train))*100))
    print("Test accuracy: {} %".format(100-np.mean(np.abs(y_pred_test-y_test))*100))
    return params

# run model
params = runNN(x_train, y_train,x_test,y_test, iters=1000)

# using keras
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform',
                         activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform',
                         activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                         activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T #reshape
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier,
                             X = x_train, y = y_train, cv = 3)
print("Accuracy mean: " + str(accuracies.mean()))
print("Accuracy std: " + str(accuracies.std()))
