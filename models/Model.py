from sklearn.metrics import matthews_corrcoef
import numpy as np
from ..debug.Logger import Logger as log
# generic Model Class
class Model:

    def __init__(self, **kwargs):
        #initializes hyper parameters 
        self.hyper = kwargs
    
    def train(self, train_x, train_y):
        # trains the model 

        pass 

    def predict(self, data_x):
        # creates prediction for each data point
        pass 

 
    def sample(self, X,y, size):
        # samples a subset of data of <size>
        size = (min(size, y.shape[0]))
        idx = np.random.choice(np.arange(len(y)), size, replace=False)
        x_sample = X[idx]
        y_sample = y[idx]
        return x_sample, y_sample