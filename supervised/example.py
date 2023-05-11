from ..models.Model import Model
from ..datasets import Banking
from sklearn.linear_model import LogisticRegression

class LogisticReg(Model):
    
    def __init__(self, **kwargs):
        # passes in params into super
        super().__init__(**kwargs)

        # default sklearn logistic regression model
        # using it as an example
        self.logi = LogisticRegression(max_iter=1000000)
    
    def train(self, train_x, train_y):
        self.logi.fit(train_x, train_y) 

    def predict(self, data_x):
        return self.logi.predict(data_x)

if __name__ == "__main__":

    #loads the testing data
    data = Banking()
    data.load()

    # runs benchmark
    LogisticReg().run_benchmarks(data)