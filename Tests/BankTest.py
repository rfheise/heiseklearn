from .Test import Test
from ..datasets import Banking


class BankTest(Test):

    def __init__(self, model):
        super().__init__(model, Banking(), "accuracy","mcc")