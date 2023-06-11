

__all__ = ["benchmarks"]


from .Accuracy import Accuracy
from .MCC import MCC 
from .MSE import MSE

benchmarks = {}
benchmarks["accuracy"] = Accuracy
benchmarks["mcc"] = MCC
benchmarks["mse"] = MSE