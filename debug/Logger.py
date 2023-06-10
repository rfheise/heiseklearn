import sys


class Logger():

    debug_bool = True

    def out(*args, **kwargs):
        print(*args, **kwargs)

    def debug(*args, **kwargs):
        if Logger.debug_bool:
            print(*args,**kwargs)

    
    def error(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)
    