from .Test import Test
from ..datasets import Titanic


class TitanicTest(Test):

    def __init__(self, model):
        super().__init__(model, Titanic(), "accuracy","mcc")