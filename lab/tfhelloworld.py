# copy/pasta fork of
# http://comet.lehman.cuny.edu/schneider/Fall17/CMP464/DemoPrograms/Beginner1A.py
import sys
import tensorflow as tf
import tempfile

log_dir=tempfile.mkdtemp(prefix="tfhelloworld")

counter = 3
a = tf.add(counter, 5)
b = a + 7
#c = b + 6
tf.summary.scalar("aa", a)
tf.summary.scalar("bb", b)
#tf.summary.scalar("cc", c)
merged = tf.summary.merge_all()

print(log_dir)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    for i in range(6):
        sumab, aa, bb = sess.run([merged, a, b])
        sys.stderr.write('results:\n\tsumab=%s\n\ta=%s\n\tb=%s\n' % (sumab, a, b))
        counter = counter + 1
        writer.add_summary(sumab, i)
        writer.flush()
    writer.close()
