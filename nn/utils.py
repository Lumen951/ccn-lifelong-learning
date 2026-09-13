import math


def sigmoid(a):
    # branch on sign so exp never overflows
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    e = math.exp(a)
    return e / (1.0 + e)
