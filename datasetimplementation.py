from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_Regression import reg
X,Y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=20)
regression=reg.linearregression()
regression.fit(X_train,Y_train)
predicted=regression.predict(X_test)

def mse(y_true,y_predicted):
    return np.mean(y_true-y_predicted)**2
mse_value=mse(Y_test,predicted)
print(mse_value)