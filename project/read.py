"""
demonstrating we can load and intepret pickle files
"""
import sys
import pickle
import numpy as np
import os

# global settings #############################################################
BATCH_SIZE = 128 # the N for the minibatches

# important constants: don't touch! ###########################################
NUM_LETTERS = 10 # size of the set of letters we're recognizing: |{a...j}|

###############################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# not really doing intersting things in this project, so just ignore optimization

class LabledDatas:
    """
    LabeledDatas describes a two-tuple of numpy.ndarrays (tensors): a dataset
    and its labels.

    Some self-mutating convenience utility is embedded, eg: toHotEncoding()
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.img_sqr_dim = self.data.shape[1]
        if len(self.data) != len(self.labels):
            raise

    def string(self):
        return "%d labels, %d data points in a %s" % (
                len(self.labels), len(self.data), self.data.shape)

    def toHotEncoding(self):
        oneHotDim = (-1, self.img_sqr_dim * self.img_sqr_dim) # "-1" means don't touch first dimension
        # takes the Array to 2 dimensional array NoDataXNofeatures
        self.data = self.data.reshape(oneHotDim).astype(np.float32)

        # to make ONE HOT ENCODING; the None adds a dimension and tricky numpy broadcasting
        self.labels = (np.arange(NUM_LETTERS) == self.labels[:,None]).astype(np.float32)

class Datas:
    """
    Datas describes three sets of data, each as LabeledDatas:
    - training set
    - validation set
    - final testing set

    Some self-mutating convenience utility is embedded, eg: toHotEncoding()
    """
    def __init__(self, training, valid, testing):
        self.training = training
        self.valid = valid
        self.testing = testing
        self.img_sqr_dim = self.training.img_sqr_dim

    def string(self):
        return "\t[training] %s\n\t[ testing] %s\n\t[ valids ] %s" % (
            self.training.string(), self.testing.string(),
            self.valid.string())

    @staticmethod
    def fromPicklePath(pklpath):
        print("loading pickle file: %s\n" % pklpath)
        with open(pklpath, 'rb') as f:
            pkdt = pickle.load(f)
            d = Datas(
                    LabledDatas(pkdt['train_dataset'], pkdt['train_labels']),
                    LabledDatas(pkdt['valid_dataset'], pkdt['valid_labels']),
                    LabledDatas(pkdt['test_dataset'], pkdt['test_labels']))
            del pkdt  # hint to help gc free up memory
            f.close()
        print("done loading pickle file;\n%s\n" % (d.string()))
        return d

    def toHotEncoding(self):
        self.training.toHotEncoding()
        self.valid.toHotEncoding()
        self.testing.toHotEncoding()

dataSets = Datas.fromPicklePath(sys.argv[1])
for i in range(15,21):
    print(dataSets.training.labels[i])
dataSets.toHotEncoding()
print("REFORMATED data to ONE HOT encoding;\n%s\n" % dataSets.string())
for i in range(15,21):
    print(dataSets.training.labels[i,:])
