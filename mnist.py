import tensorflow as tf
import numpy as np
import cv2
import os

class mnist:
    def __init__(self, sess):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.W = tf.Variable(tf.zeros([784,10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
    def predict(self):
        im = cv2.imread('./number.png', cv2.IMREAD_GRAYSCALE)
        im = im.astype(np.float32)
        im = cv2.resize(im, (28,28),interpolation=cv2.INTER_AREA)
        if(im[0][0] > 128): #바탕이 흰색에 가까우면 리버스 처리(0 - 검정, 255 - 흰)
            for i in range(28):
                for j in range(28):
                    im[i,j] = 255 - im[i,j]
        im = np.reshape(im, (-1, 784))
        res = self.sess.run(tf.argmax(self.y,1), feed_dict = {self.x: im})
        res = ''.join(map(str, res))
        return res
