import torch
import torchvision
import torchvision.datasets as datasets
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    print(train.shape, test.shape)
    # visualize(0)

    print("Train data")
    sumi = 0
    y_value = np.zeros((1, 10))
    for i in range(10):
        print("Occurence of ", i, "=", np.count_nonzero(train_labels == i))
        y_value[0, i-1] = np.count_nonzero(train_labels == i)
