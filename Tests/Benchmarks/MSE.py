from .Benchmark import Benchmark 
import numpy as np


class MSE(Benchmark):

    def run(self, model, data, **kwargs):

        # computes predictions
        pred_train, pred_test = self.extract_preds(model, data, **kwargs)

        # prints mse of predictions
        self.log(f"training mse: {self.get_mse(pred_train, data.train_y)}")
        self.log(f"testing mse: {self.get_mse(pred_test, data.test_y)}")

    def get_mse(self, preds, y):
        # computes mse
        mse = np.squeeze(np.square(y - preds).mean(axis=0))
        return mse