'''
Created 2017-09-05 10:16:55-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TrainingSet:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def costhandler(self, weightAndBias):
        """Handler for numpy's minimize() function"""
        return self.costof(weightAndBias[0], weightAndBias[1])

    def costof(self, weight, bias):
        """calculates cost using default methodology"""
        return self.costViaSquare(weight, bias)

    def costViaSquare(self, weight, bias):
        return np.square(weight*self.inputs + bias - self.labels).sum()

    def randGuessMimizes(self, debugMode=False):
        for i in range(0, 5):
            initialGuess = np.random.randn(2)
            res = minimize(self.costhandler, initialGuess, method='Nelder-Mead')
            if debugMode:
                print("\tminimized: %s\t[init guess #%d: %s]" %(res.x, i, initialGuess))

        if debugMode:
            print("context:\n\tx: %s\n\ty: %s" %(self.inputs, self.labels))
        return res # whatever the last mimizer returned

    def buildRandomTrainer(setsize=2):
        inputs = 10*np.random.randn(setsize)
        # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
        #print("x scalars, of shape: %s\n%s" % (inputs.shape, inputs))

        # labels subset of {1, -1}
        labels = 2*np.random.randint(size=setsize, low=0, high=2)-1
        #print("y scalars, of shape: %s\n%s" % (labels.shape, labels))
        return TrainingSet(inputs, labels)

def main():
    set = TrainingSet.buildRandomTrainer()
    minimd = set.randGuessMimizes()
    print("rand set's cost was %0.010f, for minimization to: %s" %
            (set.costof(minimd.x[0], minimd.x[1]), minimd.x))

if __name__ == '__main__':
    main()
