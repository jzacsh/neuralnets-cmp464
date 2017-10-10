'''
Created 2017-09-21 15:19:44-04:00
@author: jzacsh@gmail.com
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # needed for param plot='3d' below
from scipy.optimize import minimize
from matplotlib import pyplot

def buildOutputLayer(inputs, weights, bias):
    """ computes the basic `w*x + b` output layer """
    dotProd = np.dot(inputs, weights)
    return dotProd + bias

def computeOutputLayerDistances(inputs, outputs, weights, bias):
    """ computes the basic `w*x + b - y` result """
    return buildOutputLayer(inputs, weights, bias) - outputs

def costsViaSquare(inputs, outputs, weights, bias):
    return np.square(computeOutputLayerDistances(inputs, outputs, weights, bias))

def costsViaAbsVal(inputs, outputs, weights, bias):
    return np.absolute(computeOutputLayerDistances(inputs, outputs, weights, bias))

class TrainingSet:
    def __init__(self, projectTitle, inputs, labels, debugMode=False):
        self.projectName = projectTitle
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

    def costof(self, weights, bias):
        """calculates cost using default methodology"""
        return costsViaSquare(self.inputs, self.labels, weights, bias).sum()

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
        return cleanMinim(res)

    def minimize(self, initialGuess, minimAlgo='Nelder-Mead'):
        """
        the dimension of initialGuess determines the number of free variables
        the minimizer will search for

        NOTE: initial guess should contain weights, and bias as the final value
        """
        # TODO(zacsh) determine why this works; how is initialGuess inspected
        # internally??

        if self.debugMode:
            print("minimizing for a %s set of free variables [initial=%s]"%(
                initialGuess.shape, initialGuess))
        return minimize(self.costhandler, initialGuess, method=minimAlgo)

    def buildRandomTrainer(setsize=2):
        inputs = np.random.randn(setsize) * 10 # entry-wise multiply by 10
        # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
        #print("x scalars, of shape: %s\n%s" % (inputs.shape, inputs))

        # labels subset of {1, -1}
        labels = 2*np.random.randint(size=setsize, low=0, high=2)-1
        #print("y scalars, of shape: %s\n%s" % (labels.shape, labels))
        return TrainingSet(inputs, labels)

    def printReport(self, optimalWeights, optimalBias, minimOK):
        weightsStr = floatsToStr(optimalWeights)

        print("""%s set's cost was %0.05f; for minimization:
            (optimal) weights = [%s]
            (optimal) bias    = %0.04f
            minimizer success : %s\n""" % (
            self.projectName,
            self.costof(optimalWeights, optimalBias),
            weightsStr, optimalBias,
            minimOK
        ))

    def printManualLayers(self, weights, bias):
        print("""X,\t\t"w*X",\t\t"w*X"+b,\tdistance,\texpected\n%s\n"""%("="*75))
        for idx, inp in enumerate(self.inputs):
            x, y = [inp, self.labels[idx]]
            weighted = np.dot(x, weights)
            print("%s,\t\t%0.02f,\t\t%0.02f,\t\t%0.02f,\t\t%s\n" %(
                x,
                weighted,
                weighted+bias,
                weighted+bias - y,
                y))

def floatsToStr(flts):
    def printFlt(flt): return "%0.02f" % flt
    return ", ".join(map(printFlt, flts))

def cleanMinim(minimizerResult, weightCount=1):
    """returns "optimal" weight, bias, success (ie: whether vals are trustworthy)"""
    weightsAndBias = minimizerResult.x.tolist()[:weightCount+1]
    weightsAndBias.append(minimizerResult.success)
    return weightsAndBias

def generateWeightBiasSpace(weight, bias):
    sampleFrom = -3
    sampleTo = (sampleFrom) * -1
    sampleRate = 0.5

    print("\tgenerating weights & biases\n\t\t%.2f <- {weight=%0.3f, bias=%0.3f} -> %.2f @%.3f steps\n"%(
        sampleFrom, weight, bias, sampleTo, sampleRate))
    return np.meshgrid(
            np.arange(weight+sampleFrom,weight+sampleTo,sampleRate),
            np.arange(bias+sampleFrom,bias+sampleTo,sampleRate))

def learnTruthTable(binaryOp, truthTableName):
    xorinputs = np.array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]).reshape(4,2)
    xoroutputs = np.array(binaryOp).reshape(4,1)
    set = TrainingSet(truthTableName + " Truth Table", xorinputs, xoroutputs, debugMode=True)
    optimalWeight1, optimalWeight2, optimalBias, minimOK = cleanMinim(set.minimize(np.array([
        1,
        1,
        1 # bias
    ])), 2)

    set.printReport([optimalWeight1, optimalWeight2], optimalBias, minimOK)
    print("""\nmanually running reported optimal bias & weight: %0.05f
    output layer distance from input layer:
    """ % (
        costsViaSquare(
            set.inputs,
            set.labels,
            [optimalWeight1, optimalWeight2],
            optimalBias).sum())
    )
    set.printManualLayers([optimalWeight1, optimalWeight2], optimalBias)

def main():
    learnTruthTable([0, 1, 1, 0], "XOR")
    learnTruthTable([0, 1, 1, 1], "OR")
    learnTruthTable([0, 0, 0, 1], "AND")

if __name__ == '__main__':
    main()
