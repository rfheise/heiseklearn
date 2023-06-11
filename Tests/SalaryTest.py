from .Test import Test
from ..datasets import DataScience


class SalaryTest(Test):

    def __init__(self, model):
        super().__init__(model, DataScience(), "mse")