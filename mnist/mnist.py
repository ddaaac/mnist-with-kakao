import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import os

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

#선행학습
for i in range(200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys})

#save model
#saver = tf.train.Saver()
#save_path = saver.save(sess,'./saver/mnist.ckpt' )
#saver.restore(sess, './saver/mnist.ckpt')

#이미지 불러오기
path = './image/'
img_list = os.listdir(path)
data = []

for img in img_list:
    img_path = os.path.join(path, img)
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float32)
    im = cv2.resize(im, (28,28),interpolation=cv2.INTER_AREA) #사이즈 
    if(im[0][0] > 128): #바탕이 흰색에 가까우면 리버스 처리(0 - 검정, 255 - 흰)
        for i in range(28):
            for j in range(28):
                im[i,j] = 255 - im[i,j] 
    data.append([np.array(im)])


#데이터 가공
data = np.reshape(data, (-1, 784)).astype(np.float32)
data = data/255

#데이터 입력해서 각 클래스로 분류될 확률 구하기
res = sess.run(y, feed_dict = {x: data})

#출력함수
def print_result(result):
    for f in result:
        for i in range(f.shape[0]):
            print("%d(으)로 분류될 확률 : %.2f%%"%(i, 100*f[i]))

print("학습 전 확률은 다음과 같습니다.")
print_result(res)

#정답 입력받기
n = int(input("무슨 숫자였나요?"))
label = [0]*10
label[n] = 1
label = np.reshape(label, (-1, 10)).astype(np.float32)

#데이터 학습시키기
sess.run(train_step, feed_dict = {x:data, y_ : label})

#학습 후 확률 변화 값 구하기
res = sess.run(y, feed_dict = {x: data})
print("학습 후 확률은 다음과 같습니다.")
print_result(res)

