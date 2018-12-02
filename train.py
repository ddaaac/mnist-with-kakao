import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def train():
    mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot = True)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #train
    for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys})

    #save model
    saver = tf.train.Saver()
    save_path = saver.save(sess,'./saver/mnist.ckpt') 
