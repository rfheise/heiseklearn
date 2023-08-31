from .GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import SalaryTest


class LinearRegression(GLM):

    def hypothesis(self, Z):
        # hypothesis for linear regression
        # is just X @ theta
        return Z

    
if __name__ == "__main__":
    # runs Salary test benchmarks on data
    model = LinearRegression(batch_frac=1,tol=1e-5,learning_rate=.01,num_iter=10000)
    
    # kind of a bad dataset for linear regression
    # should probably replace with a better one later
    test = SalaryTest(model)
    test.run_benchmarks()

