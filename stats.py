from math import *

def erfcc(x):
    z = abs(x)
    t = 1. / (1. + 0.5 * z)
    r = t * exp(-z * z - 1.26551223 +
                t * (1.00002368 +
                t * ( .37409196 +
                t * ( .09678418 +
                t * (-.18628806 +
                t * ( .27886807 +
                t * (-1.13520398 +
                t * (1.48851587 +
                t * (-.82215223 +
                t * .17087277)))))))))
    if x >= 0.:
        return r
    else:
        return 2. -r

def normcdf(x, mu=0, sigma=1):
    t = x - mu
    y = 0.5 * erfcc(-t/(sigma * sqrt(2.0)))
    return y
