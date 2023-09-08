from ...models.Model import Model
import numpy as np
from ...Tests import BankTest,TitanicTest,SalaryTest
from .KNN import KNN

# K Nearest Neighbors for Classification

class KNNClassifier(KNN):
    
    def __init__(self, k=5, fast=False):
        super().__init__(k,False,fast)





if __name__ == "__main__":
    
    # runs Titanic Test benchmarks on data
    # not 100% on training  w/ k=1 because dataset has duplicate 
    # data points with different target values
    model = KNNClassifier(k=5)
    test = TitanicTest(model)
    test.run_benchmarks()