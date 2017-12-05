"""
demonstrating we can load and intepret pickle files
"""
import sys
import pickle
import numpy

print("loading pickle file:\n\t%s\n" % sys.argv[1])

pklpath = sys.argv[1]
with open(pklpath, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

def statsStr(data, labels):
    return "%d labels, %d data points in a %s" % (len(labels), len(data), data.shape)

print("done loading pickle file;\n\t[taining] %s\n\t[testing] %s\n\t[valids ] %s\n" % (
    statsStr(train_dataset, train_labels),
    statsStr(test_dataset, test_labels),
    statsStr(valid_dataset, valid_labels)))

