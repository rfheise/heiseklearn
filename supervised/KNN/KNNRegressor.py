from ...models.Model import Model
import numpy as np
from ...Tests import BankTest,TitanicTest,SalaryTest,Test
from ...datasets import Football
import scipy
from .KNN import KNN

# K Nearest Neighbors for Regression

class KNNRegressor(KNN):
    
    def __init__(self, k=5, fast=False):
        super().__init__(k,True,fast)





if __name__ == "__main__":

    model = KNNRegressor(k=5)
    test = SalaryTest(model)
    test = Test(model, Football())
    test.run_benchmarks()