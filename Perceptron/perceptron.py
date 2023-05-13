import numpy as np
class perceptron:
    def __init__(self,learning_rate=0.2,n_iters=1000):
        self.lr=learning_rate
        self.n_iters=n_iters
        self.weights=None
        self.bias=0
    def fit(self,X,Y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        y_=np.array([1 if y>0 else 0 for y in Y])
        for j in range(self.n_iters):
            for idx,x in enumerate(X):
                got_=self.predict(x)
                update=self.lr*(y_[idx]-got_)
                self.weights+=update*x
                self.bias+=update

    def predict(self,X):
        y_pred=np.dot(X,self.weights)+self.bias
        y_predict=np.where(y_pred>=0,1,0)
        return y_predict

        