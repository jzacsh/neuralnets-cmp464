"""
demonstrating we can load and intepret pickle files
"""
import sys
import pickle
import numpy as np

# global settings #############################################################
batch_size = 128 # the N for the minibatches
###############################################################################

# TODO: fix to at least assert these are sane values
image_size = 28
num_labels = 10
oneHotDim = (-1, image_size * image_size) # "-1": don't touch first dimension

class DataLabel:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def toString(self):
        return "%d labels, %d data points in a %s" % (
                len(self.labels), len(self.data), self.data.shape)

    def toHotEncoding(self):
        # takes the Array to 2 dimensional array NoDataXNofeatures
        self.data = self.data.reshape(oneHotDim).astype(np.float32)

        # to make ONE HOT ENCODING; the None adds a dimension and tricky numpy broadcasting
        self.labels = (np.arange(num_labels) == self.labels[:,None]).astype(np.float32)

class Datas:
    def __init__(self, training, valid, testing):
        self.training = training
        self.valid = valid
        self.testing = testing

    def string(self):
        return "\t[training] %s\n\t[ testing] %s\n\t[ valids ] %s" % (
            self.training.toString(), self.testing.toString(),
            self.valid.toString())

    @staticmethod
    def fromPicklePath(pklpath):
        print("loading pickle file: %s\n" % pklpath)
        with open(pklpath, 'rb') as f:
            pkdt = pickle.load(f)
            d = Datas(
                    DataLabel(pkdt['train_dataset'], pkdt['train_labels']),
                    DataLabel(pkdt['valid_dataset'], pkdt['valid_labels']),
                    DataLabel(pkdt['test_dataset'], pkdt['test_labels']))
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
print("REFORMATED data to ONE HOT encoding;\n%s\n" % dataSets)
for i in range(15,21):
    print(dataSets.training.labels[i,:])
