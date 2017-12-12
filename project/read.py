"""
demonstrating we can load and intepret pickle files
"""
import sys
import pickle
import numpy as np
import os
import tensorflow as tf

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

tfgraph = tf.Graph()
with tfgraph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, dataSets.img_sqr_dim * dataSets.img_sqr_dim)) #the input data
    # For the training data, we use a placeholder that will be fed at run time
    # with a training minibatch.
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LETTERS))
    tf_valid_dataset = tf.constant(dataSets.valid.data)
    tf_test_dataset = tf.constant(dataSets.testing.data)

    # Variables.
    tf_weights = tf.Variable(tf.truncated_normal([
        dataSets.img_sqr_dim * dataSets.img_sqr_dim, # the number of features
        NUM_LETTERS
    ]))
    tf_biases = tf.Variable(tf.zeros([NUM_LETTERS]))

    # Training computation.
    tf_wxb = tf.matmul(tf_train_dataset, tf_weights) + tf_biases
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_train_labels, logits=tf_wxb)) # "logits" = "unscaled log probabilities"

    # Optimizer.
    tf_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(tf_loss)

    # Predictions for the training, validation, and test data.

    # softmax: compute Pr(...) via outputs w/sigmoid & normalizing
    tf_train_prediction = tf.nn.softmax(tf_wxb)
    tf_valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, tf_weights) + tf_biases)
    tf_test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, tf_weights) + tf_biases)

with tf.Session(graph=tfgraph) as session:
    tf.global_variables_initializer().run()
    # TODO fill in missing block of code here
