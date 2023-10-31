import torch
import torchvision
import torchvision.datasets as datasets
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n+1):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


def visualize(index: int):
    plt.title((train_labels[index]))
    plt.imshow(train_data[index].reshape(28, 28), cmap=cm.binary)
    plt.show()


def check_count_of_each_label():
    y_value = np.zeros((1, 10))
    for i in range(10):
        print("Occurence of ", i, "=", np.count_nonzero(train_labels == i))
        y_value[0, i-1] = np.count_nonzero(train_labels == i)
        y_value = y_value.ravel()
        x_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.xlabel('label')
        plt.ylabel('count')
        plt.bar(x_value, y_value, 0.7, color='g')
        plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ


def softmax_backward(Z, cache):
    Z = cache
    length = 10
    dZ = np.zeros((42000, 10))
    Z = np.transpose(Z)
    for row in range(0, 42000):
        den = (np.sum(np.exp(Z[row, :])))*(np.sum(np.exp(Z[row, :])))
        for col in range(0, 10):
            sums = 0
            for j in range(0, 10):
                if (j != col):
                    sums = sums+(math.exp(Z[row, j]))

            dZ[row, col] = (math.exp(Z[row, col])*sums)/den
    dZ = np.transpose(dZ)
    Z = np.transpose(Z)

    assert (dZ.shape == Z.shape)
    return dZ

# initializing the parameters weights and bias


def initialize_parameters_deep(layer_dims):
    # np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros(layer_dims[l],
                                            layer_dims[l-1]) / np.sqrt(layer_dims[l-1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# forward propagation


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert (Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        # print("Z="+str(Z))
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
    return AL, caches

# cost function


def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = (-1) * np.sum(np.multiply(Y, np.log(AL)))
    # np.multiply(1 - Y, np.log(1 - AL)))
    # print("cost="+str(cost))
    return cost

# backward propagation


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1./m * np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        # dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    M = len(layers_dims)
    current_cache = caches[M-2]
    grads["dA"+str(M-1)], grads["dW"+str(M-1)], grads["db"+str(M-1)
                                                      ] = linear_activation_backward(dAL, current_cache, activation="softmax")  # M-1
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# upgrade function for weights and bias


def update_parameters(parameters, grads, learning_rate):
    for l in range(len_update-1):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            (learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            (learning_rate*grads["db" + str(l+1)])
    return parameters


def plot_graph(cost_plot):

    x_value = list(range(1, len(cost_plot)+1))
    # print(x_value)
    # print(cost_plot)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(x_value, cost_plot, 0., color='g')


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):  # lr was 0.009
    print("training...")
    costs = []
    cost_plot = np.zeros(num_iterations)
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost_plot[i] = cost

    plot_graph(cost_plot)
    return parameters


if __name__ == "__main__":
    mnist = datasets.MNIST(
        root='./data', download=True)
    train = pd.DataFrame()
    test = pd.DataFrame()
    if os.path.exists('./data/MNIST/raw/mnist_train.csv'):
        train = pd.read_csv("./data/MNIST/raw/mnist_train.csv")
    else:
        convert("./data/MNIST/raw/train-images-idx3-ubyte", "./data/MNIST/raw/train-labels-idx1-ubyte",
                "./data/MNIST/raw/mnist_train.csv", 60000)
        train = pd.read_csv("./data/MNIST/raw/mnist_train.csv")

    if os.path.exists('./data/MNIST/raw/mnist_test.csv'):
        test = pd.read_csv("./data/MNIST/raw/mnist_test.csv")
    else:
        convert("./data/MNIST/raw/t10k-images-idx3-ubyte", "./data/MNIST/raw/t10k-labels-idx1-ubyte",
                "./data/MNIST/raw/mnist_test.csv", 10000)
        test = pd.read_csv("./data/MNIST/raw/mnist_test.csv")
    train_labels = np.array(train.loc[:, 'label'])
    train_data = np.array(train.loc[:, train.columns != 'label'])
    # visualize(0)
    # check_count_of_each_label(train_labels)

    # d = train_data.shape[1]
    # d1 = 300
    # # Shape of W1 is given by d1 * d where d1 is 300 and d is given by 784
    # W1 = np.zeros((d1, d))

    # # print(W1.shape)
    # x1 = train_data[0]
    # # print(x1.shape, x1)
    # z1 = np.dot(W1, x1)
    # # print(z1.shape, z1)
    # a1 = sigmoid(z1)
    # print('After sigmmoid activation shape is', a1.shape)

    # W2 = np.zeros((10, d1))
    # z2 = np.dot(W2, a1)
    # # print(z2, z2.shape)
    # y_pred = softmax(z2)
    # # print(y_pred.shape)

    # y_actual = train_labels[0]
    # one_hot = np.zeros(10)
    # one_hot[y_actual] = 1
    # print(y_pred, one_hot)
    # loss = - np.dot(one_hot, np.log(y_pred))
    # print(loss)

    ###############################

    train_data = np.reshape(train_data, [784, 60000])
    train_label = np.zeros((10, 60000))
    for col in range(60000):
        val = train_labels[col]
        for row in range(10):
            if (val == row):
                train_label[val, col] = 1
    print("train_data shape="+str(np.shape(train_data)))
    print("train_label shape="+str(np.shape(train_label)))

    # n-layer model (n=3 including input and output layer)
    layers_dims = [784, 300, 10]
    len_update = len(layers_dims)
    parameters = L_layer_model(train_data, train_label, layers_dims,
                               learning_rate=0.0005, num_iterations=35, print_cost=True)
    print("training done")
