'''
Created 2017-09-05 10:16:55-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt

weight=0.5
bias=0.2

def buildTrainingSet():
    setsize=2
    xvals = 10*np.random.randn(setsize)
    # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
    #print("x scalars, of shape: %s\n%s" % (xvals.shape, xvals))

    # yvals subset of {1, -1}
    yvals = 2*np.random.randint(size=setsize, low=0, high=2)-1
    #print("y scalars, of shape: %s\n%s" % (yvals.shape, yvals))
    return (xvals, yvals)

def cost(xs, ys):
    return np.sum(np.square(weight*xs + bias - ys))

def main():
    x, y = buildTrainingSet()
    print("x: %s\ny: %s\n" %(x, y))
    print("cost:\t%s\n" %(cost(x, y)))

if __name__ == '__main__':
    main()
