import numpy as np
import math
class logistic:
    def __init__(self,lr=0.01,iter=1000):
        self.weight=None
        self.bias=None
        self.lr=lr
        self.iter=iter
    
    def model(self,X,Y):
        n_samples,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0
        for i in range(self.iter):
            linear_model=np.dot(X,self.weight)+self.bias
            y_predicted=self._sigmoid(linear_model)

            dw=(1/n_samples)*np.dot(X.T,(y_predicted-Y))
            db=(1/n_samples)*np.sum(y_predicted-Y)

            self.weight-=self.lr*dw
            self.bias-=self.lr*db
    
    def predict(self,X):
        linear_model=np.dot(X,self.weight)+self.bias
        y_predict=self._sigmoid(linear_model)
        y_predict_cs=[1 if i>0.5 else 0 for i in y_predict]
        return y_predict_cs


    def _sigmoid(self,F):
        ex=1/(1+np.exp(-F))
        return ex
