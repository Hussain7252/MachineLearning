from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from knn import knn
iris=datasets.load_iris()
X,Y=iris.data,iris.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.2,random_state=786)
clf=knn.KNN(k=1)
clf.fit(X_train,Y_train)
predictions=clf.predict(X_test)
accuracy=np.sum(predictions==Y_test)/len(Y_test)
print(accuracy)
