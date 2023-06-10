from .Benchmark import Benchmark 
import numpy as np
from sklearn.metrics import matthews_corrcoef

class MCC(Benchmark):

    def run(self, model, data):
        pred_train = model.predict(data.train_x)
        pred_test =  model.predict(data.test_x)
        self.log(f"training mcc: {self.get_mcc(pred_train, data.train_y)}")
        self.log(f"testing mcc: {self.get_mcc(pred_test, data.test_y)}")

    def get_mcc(self, preds, y):
        return matthews_corrcoef(y, preds)