from ..models.Model import Model
from ..datasets import Banking,Titanic
from ..Tests import SalaryTest, TitanicTest, BankTest
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
# example class
# mainly used to compare my models to 
# sklearn's implementation
class Example(Model):
    
    def __init__(self, **kwargs):
        # passes in params into super
        super().__init__(**kwargs)

        # default sklearn logistic regression model
        # using it as an example
        self.logi = MultinomialNB()
    
    def train(self, train_x, train_y):
        train_x = (train_x - train_x.min(axis=0))/(train_x.max(axis=0) - train_x.min(axis=0) + 1)
        self.logi.fit(train_x, train_y) 

    def predict(self, data_x):
        data_x = (data_x - data_x.min(axis=0))/(data_x.max(axis=0) - data_x.min(axis=0) + 1)
        preds = self.logi.predict(data_x)
        return np.reshape(preds,(preds.shape[0],1))

if __name__ == "__main__":

    model = Example()
    test = BankTest(model)
    test.run_benchmarks()