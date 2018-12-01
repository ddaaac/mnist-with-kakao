import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import os



def mnist():
    #mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot = True)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #y_ = tf.placeholder(tf.float32, [None, 10])

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()


    sess = tf.Session()
    sess.run(init)

    #train
    #for i in range(500):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    #sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys})


    #save model
    saver = tf.train.Saver()
    #save_path = saver.save(sess,'./saver/mnist.ckpt') 
    saver.restore(sess, './saver/mnist.ckpt')
    
    def adjust_pixel(im):
        for i in range(28):
            for j in range(28):
                im[i,j] = im[i,j] / 255
        return im

    def reverse(im):
        for i in range(28):
            for j in range(28):
                im[i,j] = 255 - im[i,j]
        return im

    im = cv2.imread('./number.png', cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float32)
    im = cv2.resize(im, (28,28),interpolation=cv2.INTER_AREA)
    if(im[0][0] > 128): #바탕이 흰색에 가까우면 리버스 처리(0 - 검정, 255 - 흰)
        im = reverse(im)
    im = adjust_pixel(im)
<<<<<<< HEAD
=======
    print(im)
>>>>>>> 05d25dd988027fa1ace9a82e988abea7031c8fd1
    im = np.array(im).astype(np.float32)

    im = np.reshape(im, (-1, 784))
    res = sess.run(tf.argmax(y,1), feed_dict = {x: im})
    res = ''.join(map(str, res))
    return res
