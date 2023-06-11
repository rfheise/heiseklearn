from .Test import Test
from ..datasets import Titanic


# Titanic data test
# small dataset used for binary classification
class TitanicTest(Test):

    def __init__(self, model):
        super().__init__(model, Titanic(), "accuracy","mcc")