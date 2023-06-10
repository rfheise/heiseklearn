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

    def run_benchmarks(self, data):
        # runs various benchmarks on trained model

        # quick metrics right now
        # will be updated in the future
        self.train(data.train_x, data.train_y)
        preds = self.predict(data.train_x)
        mcc = matthews_corrcoef(data.train_y, preds)
        acc = (preds == data.train_y).sum() / preds.size
        print("train acc: ", acc)
        print("train mcc:", mcc)
        preds = self.predict(data.test_x)
        mcc = matthews_corrcoef(data.test_y, preds)
        acc = (preds == data.test_y).sum() / preds.size
        print("test acc: ",acc)
        print("test mcc:", mcc)

    def sample(self, X,y, size):
        size = (min(size, y.shape[0]))
        idx = np.random.choice(np.arange(len(y)), size, replace=False)
        x_sample = X[idx]
        y_sample = y[idx]
        return x_sample, y_sample