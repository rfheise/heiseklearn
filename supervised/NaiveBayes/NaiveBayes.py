from ...models.Model import Model 
import numpy as np
from ...Tests import BankTest,TitanicTest,SalaryTest, Test, BankTest
from ...datasets import Pokemon

# Naive Bayes for binary classification
# Generative model that tries to estimate p(x|y)
# assumes all x are conditionally independent 
# i.e. forall x_1, x_2 p(x_1 | y) = p(x_1 | y,x_2)


class NaiveBayes(Model):

    def __init__(self, bins=10, laplace=1):
        
        # number of bins to 
        # descretize real number values into 
        self.bins = bins
        self.laplace = laplace

    def descretize(self, row, train=False):

        # descretizes row
        # store min and max of training 
        if train:
            # use training max & min for test data
            self.min = row.min(axis=0)
            self.max = row.max(axis=0)
        row = ((row - self.min)/(self.max - self.min + 1))
        row = row.T

        # descretizes into buckets
        for i in range(row.shape[0]):
            # place into bins
            bins = min(self.bins, len(np.unique(row[i])))
            row[i] = row[i] * bins 

        row = (row)//1

        return row.T
    
    def num_unique_normalized(self, row):
        
        # computes ratios of each unique value 
        d = np.unique(row, return_counts=True)[1]
        d = np.append(d, 0)

        # computes ln(phi_x_y)
        return np.log((d + self.laplace)/(d.sum() + self.laplace * d.shape[0]))

    def train(self, X, y):

        
        # number of classes
        # assumes classes are ordinal encoding
        self.classes = len(np.unique(y))
        X = X.to_numpy()
        X = self.descretize(X, train=True)

        # init params
        self.phi_y = np.unique(y, return_counts=True)[1]
        self.phi_y = np.log((self.phi_y + self.laplace)/(self.phi_y.sum() + self.laplace * self.classes))
        self.phi_y = np.reshape(self.phi_y, (self.phi_y.shape[0],1))

        # for each class compute multinomials
        self.phi_i_y = []
        
        for i in range(self.classes):
            # computes phi_i_y by only looking at X|y=i
            X_mask = X[y.reshape(-1) == i]
            self.phi_i_y.append([self.num_unique_normalized(x) for x in X_mask.T])
        
    def predict(self, X):
        
        # descretize input
        X = self.descretize(X.to_numpy())
        X = X.T

        # initialze prediction array
        preds = np.zeros((self.classes, X.shape[1]))

        #iterate over classes
        for i in range(self.classes):
            # iterate over examples
            for k in range(X.shape[0]):
                
                # iterate over features
                phis = len(self.phi_i_y[i][k])
                
                # set max array index to me max index
                X[k][X[k] >= phis] = phis - 1

                # get entire feature's phi for all examples
                preds[i] = preds[i] + self.phi_i_y[i][k][X[k].astype(np.int32)]

        # computes log prob
        preds = preds + self.phi_y
        # return max pred
        preds = np.argmax(preds, axis=0).reshape((X.shape[1],1))
        return preds
    
if __name__ == "__main__":

    model = NaiveBayes(bins=10,laplace=1)
    # test = Test(model, Pokemon())
    test = BankTest(model)
    # test = TitanicTest(model)
    test.run_benchmarks()

            

