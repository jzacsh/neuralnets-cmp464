'''
Created 2017-09-05 10:16:55-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from matplotlib import pyplot

def costsViaSquare(inputs, outputs, weight, bias):
    return np.square(weight*inputs + bias - outputs)

def costsViaAbsVal(inputs, outputs, weight, bias):
    return np.absolute(weight*inputs + bias - outputs)

class TrainingSet:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def costhandler(self, weightAndBias):
        """Handler for numpy's minimize() function"""
        return self.costof(weightAndBias[0], weightAndBias[1])

    def costof(self, weight, bias):
        """calculates cost using default methodology"""
        return costsViaSquare(self.inputs, self.labels, weight, bias).sum()

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

def sampleWeightBiasSpace(weight, bias):
    sampleFrom = -5
    sampleTo = 5
    sampleRate = 0.05

    print("\tsampling from %0.2f to %0.2f @%0.3f around weight=%0.3f, bias=%0.3f\n"%(
        sampleFrom, sampleTo, sampleRate, weight, bias))
    return pylab.meshgrid(
            np.arange(weight+sampleFrom,weight+sampleTo,sampleRate),
            np.arange(bias+sampleFrom,bias+sampleTo,sampleRate))

def main():
    set = TrainingSet.buildRandomTrainer()
    minimd = set.randGuessMimizes()
    print("rand set's cost was %0.010f, for minimization to: %s\n\tminimize success: %s\n" %
            (set.costof(minimd.x[0], minimd.x[1]), minimd.x, minimd.success))

    # grid of sampling points
    weights, biases = sampleWeightBiasSpace(minimd.x[1], minimd.x[0])
    costs = np.array([
        set.costof(w,b) for w,b in zip(np.ravel(weights),np.ravel(biases))
    ]).reshape(weights.shape)

    print("The costs (shape=%s) after sampling:\n%s\n\n"
          %(costs.shape, costs))

    # 2d-graphing machinery
    ax = pyplot.figure().add_subplot(111, projection='3d')
    ax.plot_surface(weights, biases, costs)
    ax.set_xlabel('weights')
    ax.set_ylabel('biases')
    ax.set_zlabel('costs')
    pyplot.show()

if __name__ == '__main__':
    main()
