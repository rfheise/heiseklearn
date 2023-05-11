import sys


def perr(*args, **kwargs):
    # prints to stderr instead of stdout
    print(*args, file=sys.stderr, **kwargs)