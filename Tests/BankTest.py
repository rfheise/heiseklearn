from .Test import Test
from ..datasets import Banking


# Banking data test
# large dataset used for binary classification
class BankTest(Test):

    def __init__(self, model):
        super().__init__(model, Banking(), "accuracy","mcc")