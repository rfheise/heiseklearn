from ..GLM.GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import BankTest,TitanicTest

class Perceptron(GLM):

    def hypothesis(self, Z):
        return (Z > 0).astype(int)



if __name__ == "__main__":
    # runs Titanic Test benchmarks on data
    model = Perceptron(batch_frac=1)
    test = TitanicTest(model)
    test.run_benchmarks()


