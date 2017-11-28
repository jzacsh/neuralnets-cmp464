"""
Jonathan Zacsh's solution to homework #3, Nov 14., Part II
"""
# copy/pasta fork of
# http://comet.lehman.cuny.edu/schneider/Fall17/CMP464/DemoPrograms/Beginner1A.py
import sys
import tensorflow as tf
import tempfile
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# not really doing intersting things in this lab, so just ignore optimization

def tensorOfDim(width, height, label):
    return np.random.randn(width, height).astype(np.float32)
    #alternatively, we can try to force things:
    # return tf.convert_to_tensor(matrix)
    # return tf.constant(matrix, name=label)

# instructions: sigmoid(sigmoid(sigmoid((16,8)x(8,4)) x (4,2)) x (2,1))
a = tensorOfDim(16, 8, "a")
b = tensorOfDim( 8, 4, "b")
c = tensorOfDim( 4, 2, "c")
d = tensorOfDim( 2, 1, "d")
print(a)
print(b)
print(c)
print(d)

def sigmoidAB(mA, mB, label):
    return tf.sigmoid(tf.matmul(mA, mB), label)

AB = sigmoidAB(a, b, "sigmoid-AxB")
ABC = sigmoidAB(AB, c, "sigmoid-ABxC")
ABCD = sigmoidAB(ABC, d, "sigmoid-ABCxD")
sigAll = tf.sigmoid(ABCD, "sigmoid-ABCD")

log_dir=tempfile.mkdtemp(prefix="hw3nov14")
print(log_dir)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    results = sess.run([sigAll])
    sys.stderr.write('results:\n\t%s\n' % (results))

    # note: only needed when doing a *loop* of sess.run() calls, and want to see
    # intermediary results per-loop.
    #writer.add_summary(results)

    writer.flush()
    writer.close()




# NO CODE BELOW THIS LINE IS REFERENCED/CALLED! ################################

def sigmoidMatrix(matrix, label):
    """ manually constructed sigmoid """
    denominator = tf.add(1, tf.exp(tf.multiply(-1, matrix)))
    tf.divide(1, denominator, name=label)
