from ...models.Model import Model
import numpy as np
from ...Tests import BankTest,TitanicTest,SalaryTest
import scipy

class KNN(Model):

    def __init__(self, k=5, regressor=False, fast = False):
        
        # number of nearest neighbors for classifier
        self.k = k

        # used to determine if vectorized implementation should be used
        self.fast = fast

        # boolean used to determine if classification or regressor 
        self.regressor = regressor

    def train(self, train_x, train_y):

        # save training data
        self.train_x = train_x.copy().to_numpy()
        self.train_y = train_y.copy()
    
    def predict(self, X):
        if self.fast:
            return self.predict_fast(X)
        return self.predict_low_mem(X)
        
    
    def predict_fast(self,X):

        # compute distance from each training input
        X = X.to_numpy()
        
        # subtract every row from X using every row from train
        distances =  X[:, None] - self.train_x
        
        # compute distance formula 
        # no need to square root as you are just sorting anyway
        distances = np.square(distances)
        distances = np.sum(distances, axis=2)
        

        # compute k nearest neighbors 
        top = np.argsort(distances, axis=1)
        top = top[:,:self.k]

        # replace top indicies with corresponding y values
        top_y = np.take_along_axis(self.train_y, top, axis=0)
        
        if self.regressor:
            # compute average of each row
            y_hat = np.sum(top_y, axis=1)/self.k
        else:
            # compute mode of each row
            y_hat = scipy.stats.mode(top_y,axis=1).mode

        return y_hat

    def predict_low_mem(self, X):

        # compute distance from each training input
        X = X.to_numpy()
        y_hat = np.apply_along_axis(self.k_nearest, axis=1, arr=X)
        # print(X.shape)
        # print(y_hat.shape)

        return y_hat.reshape((y_hat.shape[0],1))
    
    def k_nearest(self, x):
        x = x.reshape((1, x.shape[0]))
        # computes square of norm
        distances = self.train_x - x 
        distances = np.square(distances)
        distances = np.sum(distances, axis=1)
        
        # grabs top k index positions
        top = np.argsort(distances)
        top = top[:self.k].reshape((self.k,1))

        # finds corresponding y values for each index
        top_y = np.take_along_axis(self.train_y, top, axis=0)

        if self.regressor:
            # computes the mean
            y_hat = np.sum(top_y)/self.k
        else:
            # takes the mode
            y_hat = scipy.stats.mode(top_y, axis=0).mode

        # ensures scalar is returned
        return np.squeeze(y_hat)

