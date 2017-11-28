"""
Jonathan Zacsh's solution to homework #3, Nov 14., Part I
"""
# Per homework instructions, following lead from matlab example by professor:
#   http://comet.lehman.cuny.edu/schneider/Fall17/CMP464/Maple/PartialDerivatives1.pdf
import sys
import tensorflow as tf
import tempfile
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# not really doing intersting things in this lab, so just ignore optimization

class Differentiable:
    """ encapsulation of a function and its derivative """
    def __init__(self, label, f, d):
        self.func = f
        self.deriv = d
        self.func.name = label
        self.deriv.name = "%sDeriv" % label

# g(x) = x^4+2x-7 ; per matlab example
# g'(x) = 4x^3+2
fExFourth = Differentiable("fExFourth",
        lambda x: tf.add_n([tf.pow(x, 4), tf.multiply(2, x), -7]),
        lambda x: tf.add_n([tf.multiply(4, tf.pow(x, 3)), 2]))

tFofTwo = fExFourth.func(2)
tFofDerivTwo = fExFourth.deriv(2)

log_dir = tempfile.mkdtemp(prefix="hw3-nov14-parti")
print(log_dir)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    fOfTwo, fDerivOfTwo = results = sess.run([tFofTwo, tFofDerivTwo])
    sys.stderr.write("results:\n\tf(2)=%s\n\tf'(2)=%s\n" % (fOfTwo, fDerivOfTwo))

    # note: only needed when doing a *loop* of sess.run() calls, and want to see
    # intermediary results per-loop.
    #writer.add_summary(results)

    writer.flush()
    writer.close()
