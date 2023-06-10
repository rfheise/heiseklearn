from .GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import BankTest



class Logistic(GLM):

    def hypothesis(self, Z):
        return 1/(1 + np.exp(-Z))
    
if __name__ == "__main__":
    model = Logistic(batch_frac=1)
    test = BankTest(model)
    test.run_benchmarks()
