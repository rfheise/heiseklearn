from .GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import SalaryTest



class LinearRegression(GLM):

    def hypothesis(self, Z):
        return Z

    
if __name__ == "__main__":
    model = LinearRegression(batch_frac=1,tol=1e-5,learning_rate=.01,num_iter=10000)
    test = SalaryTest(model)
    test.run_benchmarks()

