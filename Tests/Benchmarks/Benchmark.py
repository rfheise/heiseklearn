from ...debug.Logger import Logger


# meta class I copied from stackoverflow
# automatically calls base constructor
# on all inherited classes
class meta(type):
    def __init__(cls,name,bases,dct):
        def auto__call__init__(self, *a, **kw):
            for base in cls.__bases__:
                base.__init__(self, *a, **kw)
            cls.__init__child_(self, *a, **kw)
        cls.__init__child_ = cls.__init__
        cls.__init__ = auto__call__init__

# used to create a benchmark
class Benchmark(metaclass=meta):

    def __init__(self):
        pass

    @staticmethod
    def extract_preds(model, data, **kwargs):

        if "preds_test" in kwargs and "preds_train" in kwargs:
            pred_train = kwargs["preds_train"]
            pred_test = kwargs["preds_test"]
        else:
            pred_test =  model.predict(data.test_x)
            pred_train = model.predict(data.train_x)
        return pred_train, pred_test
    def run(self, model, data, **kwargs):
        # runs bencmark on model given a specific dataset
        # assume model is pre-trained
        return None
    
    def log(self, *args, **kwargs):
        # logs to the logger
        Logger.out(*args,**kwargs)

