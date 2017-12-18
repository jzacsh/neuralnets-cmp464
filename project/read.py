"""
Learn to recognize thumbnails of hand written characters "A" through "J"
"""
import sys
import pickle
import numpy as np
import os
import tensorflow as tf
import math

# global settings #############################################################
BATCH_SIZE = 128 # the N for the minibatches

NUM_STEPS = 3001 # number of training steps to walk through

DEBUG_RATE_INVERSE = 5
# Max (minus 1) number of training steps to debug-print

REGULARIZER_EPSILON = 0.01

DEBUG_DATA_PARSING = False

###############################################################################
# important constants: don't touch! ###########################################

NUM_LETTERS = 10 # size of the set of letters we're recognizing: |{a...j}|

PICKLE_FILE = sys.argv[1]
LOG_DIR = sys.argv[2]

DEBUG_RATE_MOD = int(NUM_STEPS / DEBUG_RATE_INVERSE)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# not really doing intersting things in this project, so just ignore optimization

###############################################################################

class LabledDatas:
    """
    LabeledDatas describes a two-tuple of numpy.ndarrays (tensors): a dataset
    and its labels.

    Some self-mutating convenience utility is embedded, eg: toHotEncoding()
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.length = self.data.shape[0]
        self.img_sqr_dim = self.data.shape[1]
        if len(self.data) != len(self.labels):
            raise

    def cutBatch(self, step):
        offset = (step * BATCH_SIZE) % (self.length - BATCH_SIZE)
        batchedData = self.data[offset:(offset + BATCH_SIZE), :]
        batchedLabels = self.labels[offset:(offset + BATCH_SIZE), :]
        return batchedData, batchedLabels

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


# actual CLI logic starts here ######################

dataSets = Datas.fromPicklePath(PICKLE_FILE)
if BATCH_SIZE > dataSets.training.length:
    raise

if DEBUG_DATA_PARSING:
    for i in range(15,21): print(dataSets.training.labels[i])

dataSets.toHotEncoding()
print("REFORMATED data to ONE HOT encoding;\n%s\n" % dataSets.string())

if DEBUG_DATA_PARSING:
    for i in range(15,21): print(dataSets.training.labels[i,:])

# end of CLI logic ##################################

class Layer:
    def __init__(self, fromNodes, toNodes):
        self.w = tf.Variable(tf.truncated_normal([fromNodes, toNodes]))
        self.b = tf.Variable(tf.zeros([toNodes]))


    def wxb(self, data, label):
        """ Computes `w*x + b` on some data set, data. """
        return tf.add(tf.matmul(data, self.w), self.b, name=label)

class LayeredCake:
    """
    Encapsulates layers of a neural, preserving order, and providing access to
    start and end layers. Each layer is represented as a collection of
    TensorFlow data structures.
    """

    def __init__(self, num_feats, num_outs, num_hidden=0):
        """
        @num_feats number of feature-nodes the first layer should have.
          That is: for a given input (eg: a single image for an OCR network),
          how many distinct features are being evaluated (eg: number of pixels
          in said image).

        @num_outs number of output-nodes the final layer should have.

        @num_hidden number of hidden-layers to generate.
        """
        self.feats = num_feats
        self.outs = num_outs
        self.hidden = num_hidden

        if num_hidden == 0:
            self.layers = [Layer(self.feats, self.outs)]
        elif num_hidden == 1:
            # TODO(zacsh) finish experimenting here, and remove this in favor of
            # arbitrary num_hidden with a simple loop (and remove the exception
            # below)
            self.layers = [
                Layer(self.feats, math.ceil(self.feats/2)),
                Layer(math.ceil(self.feats/2), self.outs)
            ]
        else:
            raise NotImplementedError("have not implemented arbitrary hidden layers yet")

        self.outputs = self.layers[-1]


    def regularizers(self, epsilon):
        return epsilon * tf.add_n([tf.nn.l2_loss(lyr.w) for lyr in self.layers])


tfgraph = tf.Graph()
with tfgraph.as_default():
    num_features = dataSets.img_sqr_dim * dataSets.img_sqr_dim
    num_outputs = NUM_LETTERS

    # Input data.
    tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, num_features)) #the input data
    # For the training data, we use a placeholder that will be fed at run time
    # with a training minibatch.
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LETTERS))
    tf_valid_dataset = tf.constant(dataSets.valid.data)
    tf_test_dataset = tf.constant(dataSets.testing.data)

    cake = LayeredCake(num_features, num_outputs)

    # Training computation.
    tf_wxb = cake.outputs.wxb(tf_train_dataset, "actual")

    tf_loss = tf.reduce_mean(
            tf.reduce_mean( # "logits" = "unscaled log probabilities"
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=tf_wxb))
            + cake.regularizers(REGULARIZER_EPSILON))
    tf_optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(tf_loss)

    # Predictions for the training, validation, and test data.

    # softmax: compute Pr(...) via outputs w/sigmoid & normalizing
    tf_train_prediction = tf.nn.softmax(tf_wxb)
    tf_valid_prediction = cake.outputs.wxb(tf_valid_dataset, "valid")
    tf_test_prediction  = cake.outputs.wxb(tf_test_dataset, "test")


#############################################################
# actual training starts here ###############################

def printBatchDebug(step, cost, predic, labels):
    def debugAccuracyOf(predictions, labels):
        # predictions will be one hot encoded too and seeing if agree where 1 is
        isHighestProbabilityOnCorrectLetter = np.argmax(predictions, 1) == np.argmax(labels, 1)
        return (100.0 * np.sum(isHighestProbabilityOnCorrectLetter) / predictions.shape[0])
    pAcc = debugAccuracyOf(predic, labels)
    vAcc = debugAccuracyOf(tf_valid_prediction.eval(), dataSets.valid.labels)

    DEBUG_FMT_DOC = """\tminibatch #%d stats:
\t\t      loss: %2.10f
\t\t  accuracy: %2.3f%%
\t\tvalidation: %2.3f%%
"""
    sys.stderr.write(DEBUG_FMT_DOC % (step, cost, pAcc, vAcc))

with tf.Session(graph=tfgraph) as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    tf.global_variables_initializer().run()
    sys.stderr.write(
            "initialized & starting %d-step training [debugging every %d steps]...\n"
            % (NUM_STEPS, DEBUG_RATE_MOD))
    for step in range(NUM_STEPS):
        data, labels = dataSets.training.cutBatch(step)

        batchMapping = {tf_train_dataset: data, tf_train_labels: labels} # tensorflow-ism
        _, batchCost, batchPredictions = sess.run([  # run our actual computation
            tf_optimizer, tf_loss, tf_train_prediction
        ], feed_dict=batchMapping)

        if (step % DEBUG_RATE_MOD) == 0:
            printBatchDebug(step, batchCost, batchPredictions, labels)

        #writer.add_summary(batchCost, step)
        #writer.add_summary(batchPredictions, step)
        writer.flush()

    sys.stderr.write("TRAINING COMPLETE; last step was:\n")
    printBatchDebug(step, batchCost, batchPredictions, labels)
    writer.close()
