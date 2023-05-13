from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Logistic_Regression import logisticregression

bc=datasets.load_breast_cancer()
x,y=bc.data,bc.target
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)
regressor=logisticregression.logistic(lr=0.0001,iter=1000)
regressor.model(X_train,y_train)
predict=regressor.predict(X_test)

def accuracy(y_true,y_pred):
    accuracy=np.sum(y_true==y_pred)/(len(y_true))
    return accuracy
print(accuracy(y_test,predict))
