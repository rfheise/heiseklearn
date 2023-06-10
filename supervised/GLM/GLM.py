from ...models.Model import Model 
import numpy as np
from ...debug.Logger import Logger as log


class GLM(Model):

    def __init__(self, learning_rate=.01, num_iter=10000, tol=1e-4, batch_size=1000, batch_frac=None):
        # alpha
        self.learning_rate = learning_rate
        # number of iterations of batch gradient ascent
        self.num_iter=num_iter
        # convergence parameter of theta
        self.tol = tol
        # batch size for mini batch gradient ascent
        self.batch_size = batch_size
        # fraction of test set to use as batch size
        # can't be used with batch_size
        self.batch_frac = batch_frac

    def predict(self, X):
        return np.around(self.hypothesis(X @ self.theta))

    def train(self, X, y):
        if self.batch_frac:
            self.batch_size = int(len(X) * self.batch_frac) + 1
        self.theta = np.random.randn(X.shape[1], 1)
        theta_norm = np.linalg.norm(self.theta, axis=0)
        diff_old = None
        for i in range(self.num_iter):
            self.theta = self.theta + self.learning_rate * self.grad(X,y)
            norm = np.linalg.norm(self.theta, axis=0)
            diff = np.squeeze(norm - theta_norm)
            if diff_old != None:
                percentage_change = abs(abs(diff - diff_old)/diff_old)
                log.debug(percentage_change)
                if percentage_change < self.tol:
                    break
            diff_old = diff
            theta_norm = norm

    def grad(self, X,y):
        X,y = self.sample(X.to_numpy(),y,self.batch_size)
        y = np.reshape(y, (y.shape[0],1))
        gradient = (X.T @ (y - self.hypothesis(X @ self.theta)))/(y.shape[0])
        return gradient
            

    def hypothesis(self, Z):
        return np.zeros((Z.shape[0],1))