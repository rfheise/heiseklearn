from sklearn.metrics import matthews_corrcoef


class Model:

    def __init__(self, **kwargs):
        self.hyper = kwargs
    
    def train(self, train_x, train_y):
        pass 

    def predict(self, data_x):
        pass 

    def run_benchmarks(self, data):
        self.train(data.train_x, data.train_y)
        preds = self.predict(data.test_x)
        mcc = matthews_corrcoef(data.test_y, preds)
        acc = (preds == data.test_y).sum() / preds.size
        print("mcc:", mcc)
        print("acc:",acc)
