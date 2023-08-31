from ...models.Model import Model 
import numpy as np
from ...debug.Logger import Logger as log

#Generalized Linear Model Abstract Class

class GLM(Model):

    def __init__(self, learning_rate=.01, num_iter=10000, tol=1e-5, batch_size=1000, batch_frac=None):
        # alpha
        self.learning_rate = learning_rate

        # number of iterations of batch gradient ascent
        self.num_iter=num_iter

        # convergence parameter of theta
        self.tol = tol

        # batch size for mini batch gradient ascent
        self.batch_size = batch_size

        # fraction of train set to use as batch size
        # can't be used with batch_size
        self.batch_frac = batch_frac

    def predict(self, X):
        # just return the hypothesis on the matrix product of X & theta
        return self.hypothesis(X @ self.theta)
        
    def train(self, X, y):
        # if batch fraction is specified
        # use it to compute batch size
        if self.batch_frac:
            self.batch_size = int(len(X) * self.batch_frac) + 1

        # initialize parameters
        self.theta = np.random.randn(X.shape[1], 1)

        # computes initial norm
        # used to determine if percentage change in || theta || < tol
        theta_norm = np.linalg.norm(self.theta, axis=0)
        diff_old = None

        for i in range(self.num_iter):
            # computes new theta with grad ascent 
            self.theta = self.theta + self.learning_rate * self.grad(X,y)
            norm = np.linalg.norm(self.theta, axis=0)
            # computes difference of new and old norms
            diff = np.squeeze(norm - theta_norm)

            if diff_old != None:

                #  computes percentage change and checks to see if it is 
                # less than the tolerance
                percentage_change = abs(abs(diff - diff_old)/diff_old)
                if percentage_change < self.tol:
                    break

            diff_old = diff
            theta_norm = norm

    def grad(self, X,y):
        # performs mini batch gradient ascent 

        # samples batch from the data
        X,y = self.sample(X.to_numpy(),y,self.batch_size)
        y = np.reshape(y, (y.shape[0],1))
        
        # computes the gradient using X^t (y - h_{theta}(X))
        # divides by the number of samples to normalize
        gradient = (X.T @ (y - self.hypothesis(X @ self.theta)))/(y.shape[0])
        return gradient
            

    def hypothesis(self, Z):
        # "hypothesis" is actually dA/d eta as it is easier to compute
        return np.zeros((Z.shape[0],1))