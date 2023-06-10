

__all__ = ["benchmarks"]


from .Accuracy import Accuracy
from .MCC import MCC 


benchmarks = {}
benchmarks["accuracy"] = Accuracy
benchmarks["mcc"] = MCC