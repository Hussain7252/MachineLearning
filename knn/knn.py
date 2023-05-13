from collections import Counter 
import numpy as np
class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,Y):
        self.X_train=X
        self.Y_train=Y

    def predict(self,X):
        predicted=[self._predict(x) for x in X]
        return np.array(predicted)

    def _predict(self,x):
        distances=[self._euclidean(x,i) for i in self.X_train]
        k_indeces=np.argsort(distances)[0:self.k]
        y=[self.Y_train[i] for i in k_indeces]
        most_common=Counter(y).most_common(1)
        return most_common[0][0]

    def _euclidean(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
        


