import numpy as np
class naivebayes:
    def fit(self,X,Y):
        n_samples,n_features=X.shape
        self.classes=np.unique(Y)
        n_classes=len(self.classes)

        #init mean, var, priors
        self.mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self.variance=np.zeros((n_classes,n_features),dtype=np.float64)
        self.prior=np.zeros(n_classes,dtype=np.float64)

        for c in self.classes:
            X_c=X[c==Y]
            self.mean[c,:]=X_c.mean(axis=0)
            self.variance[c,:]=X_c.var(axis=0)
            self.prior[c]=X_c.shape[0]/float(n_samples)

    def predict(self,x):
        y_pred=[self._predict(i) for i in x]
        return y_pred
    
    def _predict(self,x):
        posteriors = []

        for idx,c in enumerate(self.classes):
            prior= np.log(self.prior[idx])
            class_conditional= np.sum(np.log(self._pdf(idx,x)))
            posterior = prior+class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx,x):
        mean=self.mean[class_idx]
        var=self.variance[class_idx]
        numerator=np.exp(-(x-mean)**2/(2*var))
        denominator=np.sqrt(2*np.pi*var)
        return numerator/denominator
