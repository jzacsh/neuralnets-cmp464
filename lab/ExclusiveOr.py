'''
Created on Aug 6, 2017

@author: bob
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid warnings about compilation
import numpy as np
import tensorflow as tf
rng = np.random

N = 4                                   # training sample size
feats = 2                               # number of input variables
inps= np.array([0,0,0,1,1,0,1,1])
#===============================================================================
# feats = 3                              # number of input variables
# inps= np.array([0,0,0,0,1,0,1,0,0,1,1,1])#add an and feature to the xor result is can train
#===============================================================================
inps= np.reshape(inps,(N,feats))
inps=inps.astype(np.float32)
print("original inputs \n",inps)
 #we will multiply on the right in tensorflow
#outps= np.array([0,1,1,0]) #exclusive or
outps= np.array([0,1,1,1])#and,or but not exclusive or work; This is what I called outps
print("original outps \n",outps)
print("shape of array \n",outps.shape)
#outps=np.reshape(outps,(4,1))
#print("shape of array 4,1 \n",outps.shape)#doesnt work in program
outps=np.reshape(outps,(N,1))
outps=outps.astype(np.float32)
print("reshaped outps \n",outps)
print("shape of array AFTER Reshaped 4, \n",outps.shape)
# generate a dataset: D = (input_values, target_class)

training_steps = 1000
#===============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Tensorflow graph
#===============================================================================
# Declare  symbolic variables

w = tf.Variable(tf.truncated_normal([feats, 1]),name='weights')# default tf.float32
b = tf.Variable(0, dtype=tf.float32,name="biases")
    
  
x = tf.placeholder(dtype=tf.float32,shape=(N,feats),name='x')#all the inputs note are four of them 

y=  tf.placeholder(dtype=tf.float32,shape=(N,1),name='y') # the corresponding correct answers





# Construct expression graph
p_1 = 1 / (1 + tf.exp(-tf.matmul(x, w) - b))   # Probability that target = 1;   4 by 1 matrix each "row" a probability
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * tf.log(p_1) - (1-y) * tf.log(1-p_1) # Cross-entropy loss function; Manipulation of maximum likelihood taking logs and negative sign 
#to minimize result; 
#-- note when I didnt make the y (N,1) totally screwed up per usual gave 4X4 for entropy

cost = tf.reduce_mean(xent) + 0.01 * tf.reduce_mean(tf.square(w))# The cost to minimize with an L2 regularization
 
#optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

derivW= tf.gradients(cost,w,name='derivW')
derivb= tf.gradients(cost,b,name='derivb')     # Compute the gradient of the cost
    

 
newW=w-.1*derivW[0] #I needed reduce sum before but here they both should be vectors unless I have to take deivW[0] or something
trainw=tf.assign(w,newW)
newb=b-.1*derivb[0]
trainb=tf.assign(b,newb)

#===============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# End of Creating Graph
#===============================================================================


# Compile
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong


# Train
for i in range(training_steps):
    feed_dict= {x:inps, y:outps}
    #_, ww,bb,p1,xentropy = sess.run([optimizer, w, b,p_1,xent],feed_dict=feed_dict)
    bb,ww,dw,db,tw,tb=sess.run([b,w,derivW,derivb,trainw,trainb],feed_dict=feed_dict )# want trainw and trainb so that the assign is run
    
        #print("\n\n new w b \n",ww," \n\n ",bb)
        #print("\n\n output ",p1,"shape ",p1.shape)
        #print("\n\n entropy",xentropy)
        #print("\n\n\n new dw db \n",dw," \n \n",db)


print("\n\n\n\n\n Final model:  b values \n\n")
print(bb)
print("target values for D:")
print(outps)
print("prediction on D:")
feed_dict= {x:inps}
pred=sess.run(prediction,feed_dict=feed_dict)
print(pred)
print("correct answers when zero from training ")
print(pred.astype(int)-outps)

""" Homework

1. change using a minimizing of L2 difference. This is normal regression. What logical expressions can you learn.
2. See if regularization makes any difference
3.  Add a couple of random digits to the input for XOR of first two digits.  See if you can fit this situation with 4 test samples.
Can you test to see if it generalizes.What if you have more than 4 test samples? (Look at my Theano/XORRandomExtraDimensions.py)
4. Can you look at derivatives to tell when you are close to a min and stop the routine
"""
