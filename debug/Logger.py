import sys


# class used to log message
class Logger():

    debug_bool = True

    def out(*args, **kwargs):
        # logs to standard out
        # used for showing message to user
        print(*args, **kwargs)

    def debug(*args, **kwargs):
        if Logger.debug_bool:
            # logs to standard out
            # used for showing dubug messages to dev
            print(*args,**kwargs)

    
    def error(*args, **kwargs):
        # logs to standard error
        # used for error messages
        print(*args, file=sys.stderr, **kwargs)
    