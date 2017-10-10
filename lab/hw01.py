'''
Created 2017-09-21 15:19:44-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # needed for param plot='3d' below
from scipy.optimize import minimize
from matplotlib import pyplot

def costsViaSquare(inputs, outputs, weight, bias):
    dotProd = np.dot(weight, inputs)
    return np.square(dotProd + bias - outputs)

def costsViaAbsVal(inputs, outputs, weight, bias):
    return np.absolute(np.dot(weight, inputs) + bias - outputs)

class TrainingSet:
    def __init__(self, inputs, labels, debugMode=False):
        self.debugMode = debugMode
        self.inputs = np.array(inputs)
        self.labels = np.array(labels)

        if self.debugMode:
            print("constructing trainer given %dx%x inputs, %dx%d expected output matrices" % (
                self.inputs.shape[0], self.inputs.shape[1],
                self.labels.shape[0], self.labels.shape[1]))

    def costhandler(self, weightAndBias):
        """Handler for numpy's minimize() function"""
        return self.costof(weightAndBias[0], weightAndBias[1])

    def costof(self, weight, bias):
        """calculates cost using default methodology"""
        return costsViaSquare(self.inputs, self.labels, weight, bias).sum()

    def randGuessMimizes(self):
        """returns "optimal" weight, bias, success (ie: whether vals are trustworthy)"""
        if self.debugMode:
            print("randomly guessing & minimizing.... Knowns are\n\tx: %s\n\ty: %s" %(self.inputs, self.labels))

        for i in range(0, 5):
            ithGuess = np.random.randn(2) # two guesses: one for weight, one for bias
            res = self.minimize(ithGuess)
            if self.debugMode:
                print("\tminimized: %s\t[init guess #%d: %s]" %(res.x, i, ithGuess))

        # whatever the last mimizer returned
        return [ res.x[0], res.x[1], res.success ]

    def minimize(self, initialGuess, minimAlgo='Nelder-Mead'):
        return minimize(self.costhandler, initialGuess, method=minimAlgo)

    def buildRandomTrainer(setsize=2):
        inputs = np.random.randn(setsize) * 10 # entry-wise multiply by 10
        # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
        #print("x scalars, of shape: %s\n%s" % (inputs.shape, inputs))

        # labels subset of {1, -1}
        labels = 2*np.random.randint(size=setsize, low=0, high=2)-1
        #print("y scalars, of shape: %s\n%s" % (labels.shape, labels))
        return TrainingSet(inputs, labels)

def generateWeightBiasSpace(weight, bias):
    sampleFrom = -3
    sampleTo = (sampleFrom) * -1
    sampleRate = 0.5

    print("\tgenerating weights & biases\n\t\t%.2f <- {weight=%0.3f, bias=%0.3f} -> %.2f @%.3f steps\n"%(
        sampleFrom, weight, bias, sampleTo, sampleRate))
    return np.meshgrid(
            np.arange(weight+sampleFrom,weight+sampleTo,sampleRate),
            np.arange(bias+sampleFrom,bias+sampleTo,sampleRate))

def main():
    xorinputs = np.meshgrid(
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1])
    xoroutputs = np.meshgrid([0], [1], [1], [0])
    set = TrainingSet(xorinputs, xoroutputs, debugMode=True)
# TODO(zacsh) figure out exactly what professor wants us to do with the xor
# table...
#   optimalWeight, optimalBias, minimOK = set.minimize(np.array([1, 1]))

    print("""rand set's cost was %0.05f
    for minimization with: (optimal) weight=%0.04f, (optimal) bias=%0.04f
    [minimize success: %s]""" % (
        set.costof(optimalWeight, optimalBias),
        optimalWeight, optimalBias,
        minimOK
    ))

if __name__ == '__main__':
    main()
