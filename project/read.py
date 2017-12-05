"""
demonstrating we can load and intepret pickle files
"""
import sys
import pickle
import numpy as np

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
    f.close()

def statsStr(data, labels):
    return "%d labels, %d data points in a %s" % (len(labels), len(data), data.shape)

print("done loading pickle file;\n\t[taining] %s\n\t[testing] %s\n\t[valids ] %s\n" % (
    statsStr(train_dataset, train_labels),
    statsStr(test_dataset, test_labels),
    statsStr(valid_dataset, valid_labels)))

for i in range(15,21):
    print(train_labels[i])

image_size = 28
num_labels = 10
oneHotDim = (-1, image_size * image_size) # "-1": don't touch first dimension

def reformat(dataset, labels):
    # takes the Array to 2 dimensional array NoDataXNofeatures
    dataset = dataset.reshape(oneHotDim).astype(np.float32)

    # to make ONE HOT ENCODING; the None adds a dimension and tricky numpy broadcasting
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("REFORMATED data to ONE HOT encoding;\n\t[taining] %s\n\t[testing] %s\n\t[valids ] %s\n" % (
    statsStr(train_dataset, train_labels),
    statsStr(test_dataset, test_labels),
    statsStr(valid_dataset, valid_labels)))

for i in range(15,21):
    print(train_labels[i,:])
