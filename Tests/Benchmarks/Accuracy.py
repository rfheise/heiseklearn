from .Benchmark import Benchmark 
import numpy as np

class Accuracy(Benchmark):

    def run(self, model, data):
        pred_train = model.predict(data.train_x)
        pred_test =  model.predict(data.test_x)
        self.log(f"training accuracy: {self.get_accuracy(pred_train, data.train_y)}")
        self.log(f"testing accuracy: {self.get_accuracy(pred_test, data.test_y)}")

    def get_accuracy(self, preds, y):
        num_correct = np.squeeze(np.sum(y == preds)) 
        acc = num_correct/len(preds)
        return acc * 100