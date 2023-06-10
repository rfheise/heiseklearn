from ...debug.Logger import Logger



class meta(type):
    def __init__(cls,name,bases,dct):
        def auto__call__init__(self, *a, **kw):
            for base in cls.__bases__:
                base.__init__(self, *a, **kw)
            cls.__init__child_(self, *a, **kw)
        cls.__init__child_ = cls.__init__
        cls.__init__ = auto__call__init__

class Benchmark(metaclass=meta):

    def __init__(self):
        pass

    def run(self, model, data):
        # assume model is pre-trained
        return None
    
    def log(self, *args, **kwargs):
        Logger.out(*args,**kwargs)

