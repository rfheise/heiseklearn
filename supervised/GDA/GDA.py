from ...models.Model import Model
import numpy as np
from ...Tests import BankTest,TitanicTest
# Gaussian Discriminant Analysis
# Generative Model that tries to fit 
# a Gaussian to p(x|y)
class GDA(Model):

    def __init__(self, noise=.5):
        self.noise = noise
        pass 

    def train(self, X,y):
        
        # perform MLE to find maximizing params

        X = X.to_numpy()
        # add gaussian noise to data
        # kind of cheap but makes it so I don't have 
        # to force columns to be linearly independent
        X = np.random.normal(loc=0,scale = self.noise,size=X.shape) + X
        
        # find phi for p(y)
        self.phi = self.mean_y(y)

        # find sigma, mu1, mu0 for gaussians
        self.mu0 = self.mean_x(0, X, y)
        self.mu1 = self.mean_x(1, X, y)
        self.sigma = self.covariance_matrix(X, y, self.mu0, self.mu1)

    def mean_x(self,y_val, X, y):
        # using xor to determine which 
        # y to keep 
        # more efficient than y^y * (1- y)^(1 - y)
        # but same operation
        y_keep = np.logical_xor(y, y_val)
        y_keep = np.logical_xor(y_keep, 1).astype(np.int0)
        mu = np.sum(X * y_keep, axis=0)/np.sum(y_keep)
        return mu.reshape((1, mu.shape[0]))
    
    def mean_y(self,y):
        return np.sum(y)/len(y)
    
    def covariance_matrix(self, X, y, mu0, mu1):
        # subtract appropriate mu
        X_1 = (X - mu1) * y
        X_0 = (X - mu0) * np.logical_xor(y,1).astype(np.int8)
        X = (X_0 + X_1)
        return (X.T @ X)/X.shape[0]

    def multivariate_gaussian(self,X, mu, sigma):
        # compute gaussian 
        X = X.to_numpy()
        constant = 1/((2 * np.pi) ** (X.shape[1]/2) * np.linalg.det(sigma)**(1/2))
        inverse_sig = np.linalg.inv(sigma)
        exp = (X - mu) @ inverse_sig 
        exp = exp.T * (X - mu).T
        exp = np.sum(exp, axis= 0)
        data = constant * np.exp(-.5 * exp.T)
        return data
    
    def predict(self, X):

        # calculate p(y=1|x)
        p_y_1_x = self.multivariate_gaussian(X,self.mu1, self.sigma) * self.phi
        # calculate p(y=0|x)
        p_y_0_x = self.multivariate_gaussian(X,self.mu0, self.sigma) * (1 - self.phi)

        # use whichever is larger for prediction
        preds = (p_y_1_x > p_y_0_x).astype(np.int8)
        return preds.reshape(preds.shape[0], 1)

if __name__ == "__main__":
    
    # runs Titanic Test benchmarks on data
    model = GDA()
    test = BankTest(model)
    test.run_benchmarks()