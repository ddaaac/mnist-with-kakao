import tensorflow as tf
import sys
import os
from mnist import mnist

model = mnist

def _setup_():
    global model
    
    sess = tf.Session()
    
    model = mnist(sess)
    
    saver = tf.train.Saver()
    
    saver.restore(sess, './saver/mnist.ckpt')
    
def _get_response_():   
    return model.predict()