from ..datasets import Titanic
from .Benchmarks import benchmarks
from ..debug.Logger import Logger as log

class Test:
    
    def __init__(self, model, data, *benchmarks):
        self.model = model 
        self.data = data
        self.benchmarks = benchmarks

    def run_benchmarks(self):
        
        #load data into memory 
        self.data.load()

        self.model.train(self.data.train_x, self.data.train_y)

        for test in self.benchmarks:
            if test not in benchmarks:
                log.error(f"Benchmark {test} not found!")
                continue 
            benchmark = benchmarks[test]()
            benchmark.run(self.model, self.data)
                
        