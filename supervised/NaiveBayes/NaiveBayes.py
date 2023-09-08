from ...models.Model import Model 
import numpy as np
from ...Tests import BankTest,TitanicTest,SalaryTest, Test, BankTest
from ...datasets import Pokemon

# Naive Bayes for binary classification
# Generative model that tries to estimate p(x|y)
# assumes all x are conditionally independent 
# i.e. forall x_1, x_2 p(x_1 | y) = p(x_1 | y,x_2)

# not the best implementation as I don't use much 
# vectorization 
# seems hard to optimize with vectorization
# as a lot of operations that are bad for mem caching
# but could benefit from running in parallel
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
            print("computing phi")
            # computes phi_i_y by only looking at X|y=i
            X_mask = X[y.reshape(-1) == i]
            self.phi_i_y.append([self.num_unique_normalized(x) for x in X_mask.T])
        
    def predict(self, X):
        
        # descretize input
        X = self.descretize(X.to_numpy())

        # initialze prediction array
        preds = np.zeros((self.classes, X.shape[0]))
        print("generating predictions")
        #iterate over classes
        for i in range(self.classes):
            # iterate over examples
            for j in range(X.shape[0]):
                # if j % 100:
                    # print(j)
                # iterate over features
                for k in range(X.shape[1]):
                    phis = len(self.phi_i_y[i][k])
                    # if feature not seen label as unknown
                    if phis <= int(X[j][k]):
                        preds[i][j] += self.phi_i_y[i][k][phis - 1]
                    else:
                        # else use calculated phi
                        preds[i][j] += self.phi_i_y[i][k][int(X[j][k])]
        # computes log prob
        preds = preds + self.phi_y
        # return max pred
        preds = np.argmax(preds, axis=0).reshape((X.shape[0],1))
        return preds
    
if __name__ == "__main__":

    model = NaiveBayes(bins=10,laplace=1)
    # test = Test(model, Pokemon())
    test = BankTest(model)
    # test = TitanicTest(model)
    test.run_benchmarks()

            

