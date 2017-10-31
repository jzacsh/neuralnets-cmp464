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

def logisticSigmoid(x):
    return 1/(1+np.exp(-x))

def crossEntropyParts(Ys, sigmoidOutputs):
    oneMinus = (1-Ys)*np.log(1 - sigmoidOutputs)
    yTimes = Ys*np.log(sigmoidOutputs)
    return np.sum(-1 * oneMinus) + np.sum(-1 * yTimes)

def costsViaCrossEntropy(inputs, outputs, weights, bias):
    layerOut = computeOutputLayerDistances(inputs, outputs, weights, bias)
    return crossEntropyParts(outputs, logisticSigmoid(layerOut))

class TrainingSet:
    def __init__(self, projectTitle, inputs, labels, debugMode=False):
        self.projectName = projectTitle
        self.debugMode = debugMode
        self.inputs = np.array(inputs)
        self.labels = np.array(labels)
        self.weightCount = self.inputs.shape[1] if self.inputs.shape[1] else 1

        if self.debugMode:
            print("constructing trainer given %dx%x inputs, %dx%d expected output matrices" % (
                self.inputs.shape[0], self.inputs.shape[1],
                self.labels.shape[0], self.labels.shape[1]))

    def costhandler(self, weightsAndBias):
        """Handler for numpy's minimize() function"""
        wbli = weightsAndBias.tolist()
        weights = np.array(wbli[:self.weightCount]).reshape(self.weightCount, 1)
        bias = wbli[self.weightCount]
        if self.debugMode:
            print("[dbg] ...minimizing... weights=[%0.02f, %0.02f], bias=%0.03f"%(
                weights[0, 0], weights[1, 0], bias))
        return self.costof(weights, bias)

    def costof(self, weights, bias):
        """calculates cost using default methodology"""
        return costsViaCrossEntropy(self.inputs, self.labels, weights, bias).sum()

    def randGuessMimizes(self):
        """returns "optimal" weight, bias, success (ie: whether vals are trustworthy)"""
        if self.debugMode:
            print(
                "random guessing & minimizing.... Knowns are\n\tx: %s\n\ty: %s"
                %(self.inputs, self.labels))

        for i in range(0, 5):
            ithGuess = np.random.randn(2) # two guesses: one for weight, one for bias
            res = self.minimize(ithGuess)
            if self.debugMode:
                print("\tminimized: %s\t[init guess #%d: %s]" %(res.x, i, ithGuess))

        # whatever the last mimizer returned
        return cleanMinim(res, self.weightCount)

    def minimize(self, initialGuess, minimAlgo='Nelder-Mead'):
        """
        the dimension of initialGuess determines the number of free variables
        the minimizer will search for

        NOTE: initial guess should contain weights, and bias as the final value
        """
        if self.debugMode:
            print("... scipy.optimize minimizing (%s) on %s Wv,b free vars [initial=%s]"%(
                minimAlgo, initialGuess.shape, initialGuess))
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

        print("""RESULT: "%s" set's cost was %0.05f; for minimization:
            (optimal) weights = [%s]
            (optimal) bias    = %0.04f
            minimizer success : %s\n""" % (
            self.projectName,
            self.costof(optimalWeights, optimalBias),
            weightsStr, optimalBias,
            minimOK
        ))

    def printManualLayers(self, weights, bias, resultCaster):
        print("""X,\t\t"w*X",\t\t"w*X"+b,\tdistance,\texpected,\tanswer\n%s\n"""%("="*79))
        for idx, inp in enumerate(self.inputs):
            x, y = [inp, self.labels[idx]]
            weighted = np.dot(x, weights)
            wxPlusB = weighted+bias
            cost = crossEntropyParts(y, logisticSigmoid(wxPlusB - y))
            print("%s,\t\t%0.02f,\t\t%0.02f,\t\t%0.02f,\t\t%s\t\t%0.1d\n" %(
                x,
                weighted,
                wxPlusB,
                cost,
                y,
                resultCaster(wxPlusB)))

def floatsToStr(flts):
    def printFlt(flt): return "%0.02f" % flt
    return ", ".join(map(printFlt, flts))

def cleanMinim(minimizerResult, weightCount=1):
    """returns "optimal" weight, bias, success (ie: whether vals are trustworthy)"""
    results = minimizerResult.x.tolist()
    weights = results[:weightCount]
    bias = results[weightCount]
    return [weights, bias, minimizerResult.success]

def generateWeightBiasSpace(weight, bias):
    sampleFrom = -3
    sampleTo = (sampleFrom) * -1
    sampleRate = 0.5

    print("\tgenerating weights & biases\n\t\t%.2f <- {weight=%0.3f, bias=%0.3f} -> %.2f @%.3f steps\n"%(
        sampleFrom, weight, bias, sampleTo, sampleRate))
    return np.meshgrid(
            np.arange(weight+sampleFrom,weight+sampleTo,sampleRate),
            np.arange(bias+sampleFrom,bias+sampleTo,sampleRate))

def learnTruthTable(binaryOp, truthTableName, resultCaster):
    print("\nLearning to produce: %s...\n" % (binaryOp))
    xorinputs = np.array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]).reshape(4,2)
    xoroutputs = np.array(binaryOp).reshape(4,1)
    set = TrainingSet(
            truthTableName + " Truth Table",
            xorinputs,
            xoroutputs, debugMode=False)
    optimalWeights, optimalBias, minimOK = cleanMinim(
            set.minimize(np.array([1, 1, 1])), set.weightCount)

    set.printReport(optimalWeights, optimalBias, minimOK)
    print("""Manually running said bias & weights, with cost = %0.05f (via sum-of-squares)
    """ % (costsViaCrossEntropy(
        set.inputs,
        set.labels,
        optimalWeights,
        optimalBias).sum()))
    set.printManualLayers(optimalWeights, optimalBias, resultCaster)

def positive(val):
    return 0 if val < 0 else 1

def main():
    learnTruthTable([0, 1, 1, 0], "XOR", lambda wxPlusB: positive(wxPlusB))
    learnTruthTable([0, 1, 1, 1], "OR",  lambda wxPlusB: positive(wxPlusB))
    learnTruthTable([0, 0, 0, 1], "AND", lambda wxPlusB: positive(wxPlusB))

if __name__ == '__main__':
    main()
