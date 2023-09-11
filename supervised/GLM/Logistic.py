from .GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import BankTest,TitanicTest, Test 
from ...datasets import Pokemon, Football



class Logistic(GLM):

    def hypothesis(self, Z):
        # hypothesis for logisitc is just sigmoid func
        return 1/(1 + np.exp(-Z))
    
    def predict(self, X):
        # rounds hypothesis to nearest whole number (0,1)
        return np.around(super().predict(X))
    
if __name__ == "__main__":
    
    # runs Pokemon Test benchmarks on data
    model = Logistic(batch_frac=1)
    test = Test(model, Football())
    # test = BankTest(model)
    test.run_benchmarks()
    print(model.theta)

