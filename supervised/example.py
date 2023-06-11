from ..models.Model import Model
from ..datasets import Banking,Titanic
from ..Tests import SalaryTest
from sklearn.linear_model import LinearRegression

class Example(Model):
    
    def __init__(self, **kwargs):
        # passes in params into super
        super().__init__(**kwargs)

        # default sklearn logistic regression model
        # using it as an example
        self.logi = LinearRegression()
    
    def train(self, train_x, train_y):
        self.logi.fit(train_x, train_y) 

    def predict(self, data_x):
        return self.logi.predict(data_x)

if __name__ == "__main__":

    model = Example()
    test = SalaryTest(model)
    test.run_benchmarks()