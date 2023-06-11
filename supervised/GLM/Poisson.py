from .GLM import GLM
import numpy as np
from ...debug.Logger import Logger as log
from ...Tests import BankTest,TitanicTest,FertilityTest



class Poisson(GLM):

    def hypothesis(self, Z):
        # hypothesis for poisson is e^(X @ theta)
        return np.exp(Z)
    


