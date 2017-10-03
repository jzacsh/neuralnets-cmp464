'''
Created 2017-09-05 10:16:55-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # needed for param plot='3d' below
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
            ithGuess = np.random.randn(2) # two guesses: one for weight, one for bias
            res = minimize(self.costhandler, ithGuess, method='Nelder-Mead')
            if debugMode:
                print("\tminimized: %s\t[init guess #%d: %s]" %(res.x, i, ithGuess))

        if debugMode:
            print("context:\n\tx: %s\n\ty: %s" %(self.inputs, self.labels))
        return res # whatever the last mimizer returned

    def buildRandomTrainer(setsize=2):
        inputs = np.random.randn(setsize) * 10 # entry-wise multiply by 10
        # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
        #print("x scalars, of shape: %s\n%s" % (inputs.shape, inputs))

        # labels subset of {1, -1}
        labels = 2*np.random.randint(size=setsize, low=0, high=2)-1
        #print("y scalars, of shape: %s\n%s" % (labels.shape, labels))
        return TrainingSet(inputs, labels)

def generateWeightBiasSpace(weight, bias):
    sampleFrom = -5
    sampleTo = 5
    sampleRate = 0.05

    print("\tsampling from %0.2f to %0.2f @%0.3f around weight=%0.3f, bias=%0.3f\n"%(
        sampleFrom, sampleTo, sampleRate, weight, bias))
    return np.meshgrid(
            np.arange(weight+sampleFrom,weight+sampleTo,sampleRate),
            np.arange(bias+sampleFrom,bias+sampleTo,sampleRate))

def main():
    set = TrainingSet.buildRandomTrainer()
    minimd = set.randGuessMimizes()
    optimalWeight = minimd.x[0]
    optimalBias = minimd.x[1]

    print("rand set's cost was %0.010f, for minimization to: %s\n\tminimize success: %s\n" %
            (set.costof(optimalWeight, optimalBias), minimd.x, minimd.success))

    #TODO fix this line + scatterplot graphing section; totally broken
    # plot coordinates of (input,costs) and graph the minimized result
#   ax = pyplot.figure().add_subplot(111, projection='3d')
#
#   # graph a line of our minimized (optimal) weight + bias:
#   #TODO: graph a line on pyplot; subplot??
#   #mx+b form is: m=optimalWeight, b=optimalBias
#
#   #TODO: graph plot of dots, whose x,y coords are:
#   #x = set.inputs
#   #y = set.labels
#
#   #TODO: place above two plots on single x-y plane:
    costs = costsViaSquare(set.inputs, set.labels, optimalWeight, optimalBias)
    pyplot.scatter(set.inputs, costs)
    pyplot.plot(set.inputs, optimalWeight*set.inputs+optimalBias, '-')


    # grid of sampling points
    weights, biases = generateWeightBiasSpace(optimalWeight, optimalBias)
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
