from Perceptron import perceptron
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

X,Y=datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=786)
p=perceptron.perceptron(learning_rate=0.01,n_iters=1000)
p.fit(X_train,Y_train)
print(accuracy(Y_test,p.predict(X_test)))

