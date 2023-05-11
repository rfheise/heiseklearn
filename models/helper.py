import sys


def perr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)