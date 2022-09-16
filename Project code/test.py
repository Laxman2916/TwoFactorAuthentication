import pandas as pd
import numpy as np
import os


def getID(name):
    arr = name.split(".")
    arr = arr[0].split("_")
    return int(arr[1])


dataset = 'ECG-Dataset'
X = []
Y = []

for root, dirs, directory in os.walk(dataset):
    for j in range(len(directory)):
        name = getID(directory[j])
        print(str(name)+" "+root+"/"+directory[j])
        dataset = pd.read_csv(root+"/"+directory[j],header=None)
        dataset = dataset.values
        X.append(dataset)
        Y.append(name)

X = np.asarray(X)
Y = np.asarray(X)
print(X)
print(Y)
print(X.shape)
print(Y.shape)

XX = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
print(XX.shape)
               
