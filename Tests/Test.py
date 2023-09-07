from ..datasets import DataSet
from .Benchmarks import benchmarks
from ..debug.Logger import Logger as log
import numpy as np

# Test class
# used to run benchmarks on a dataset
class Test:
    
    def __init__(self, model, data, *benchmarks):
        # model to run benchmarks on
        self.model = model 

        # dataset to run benchmarks on
        self.data = data

        # list of benchmarks to run
        self.benchmarks = benchmarks

    def generate_benchmarks(self):
        
        # automatically generate benchmarks 
        # depending on dataset target type
        # definitely needs work but good for now 

        if len(np.unique(self.data.train_y)) == 2:
            self.benchmarks = ['accuracy','mcc']
        else:
            self.benchmarks = ['mse']



    def run_benchmarks(self):

        #load data into memory 
        self.data.load()
        
        # if benchmarks empty populate benchmarks with 
        # default ones to match dataset
        if self.benchmarks == ():
            self.generate_benchmarks()

        # trains model
        self.model.train(self.data.train_x, self.data.train_y)

        for test in self.benchmarks:
            # checks to see if benchmark in list of 
            # supported benchmarks see __init__.py of Benchmarks
            if test not in benchmarks:
                log.error(f"Benchmark {test} not found!")
                continue 

            # initializes & runs benchmark
            benchmark = benchmarks[test]()
            benchmark.run(self.model, self.data)
                
        