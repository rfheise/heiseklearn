from .Benchmark import Benchmark 
import numpy as np

class Accuracy(Benchmark):

    def run(self, model, data, **kwargs):

        # computes predictions
        pred_train, pred_test = self.extract_preds(model, data, **kwargs)
        
        # prints accuracy of predictions
        self.log(f"training accuracy: {self.get_accuracy(pred_train, data.train_y)}")
        self.log(f"testing accuracy: {self.get_accuracy(pred_test, data.test_y)}")

    def get_accuracy(self, preds, y):
        # computes accuracy 
        num_correct = np.squeeze(np.sum(y == preds)) 
        acc = num_correct/len(preds)
        return acc * 100