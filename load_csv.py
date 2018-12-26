import tensorflow as tf
import numpy as np
import pandas as pd
import random


'''
    입력값(x) 제한 : 숫자
    라벨값(label) 제한 : 분류할 class의 갯수가 n 이라면 (0,1,2, ... n-1) 중 하나의 숫자
    csv 파일 형식 : 행 - 각 테스트 케이스, 열 - 속성
                   첫행 - 분류할 속성의 이름(ex. 수학점수, 영어점수, 국어점수 등)
                   열은 구분 index 없이 바로 속성값을 입력해야함

'''

attri_num = 5  #사용할 속성 수 입력 - 읽어올 수도 있는데 이렇게 직접 쓰면 좀 더 이해가 되지 않을까싶어서 직접 입력하도록 함.
class_num = 2   #사용할 분류 수 입력 - 마찬가지


train_data = []
label_data = []


read_file = pd.read_csv('./weatherAUS.csv', encoding='euc-kr') #파일 읽기
for col in read_file: #음수값, 결측값 제거
    read_file = read_file[read_file[col] >= 0]
read_file = read_file.reset_index(drop=True) #지운 파일들 index 초기화
label_read = read_file[['label']].values #라벨 읽기
read_file = read_file.drop('label',1)    #라벨 열 삭제

attribute_max = []
maxidx = read_file.abs().idxmax()
#해당 속성이 갖는 최대값으로 나누기 -> 0~1사이로 보정
for idx, col in enumerate(read_file):
    an_attribute_max = float(read_file.loc[maxidx[idx],col]) 
    if (an_attribute_max != 0):
        read_file[col] = read_file[col] / an_attribute_max
    attribute_max.append(an_attribute_max) #속성별 최댓값 저장 -> 나중에 test case 입력 받고 그 값들도 나눠줘야되니깐.


#train data와 label 읽어오기
for idx, row in read_file.iterrows():
    train_data.append(row.values)       #한 케이스에 대하여 속성 값들을 train_data에 추가
    one_hot = [0]*class_num             #label에 대하여 one hot encoding 해주기
    one_hot[label_read[idx][0]] = 1
    label_data.append(one_hot)

sample_size = len(train_data)           #train_data의 크기를 알아서 읽어옴

#학습 구성
x = tf.placeholder(tf.float32, [None, attri_num])
W = tf.Variable(tf.zeros([attri_num,class_num]))
b = tf.Variable(tf.zeros([class_num]))
y = tf.nn.softmax(tf.matmul(x, W)+ b)

y_ = tf.placeholder(tf.float32, [None, class_num])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for idx in range(500):
    numlist = random.sample(range(0,sample_size),100) #0~size중의 숫자 중에서 랜덤으로 100개를 추출 - 100개 이상의 데이터는 들어가야됨
    batch_xs = []
    batch_ys = []
    for i in numlist:
        batch_xs.append(train_data[i])
        batch_ys.append(label_data[i])
    sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys})

#save model
#saver = tf.train.Saver()
#save_path = saver.save(sess,'./saver/save.ckpt')
#saver.restore(sess,'./saver/save.ckpt')

#test data 입력
data = []
a_data = []
for i in range(attri_num):
    n = float(input("{}번째 속성 값 입력 : ".format(i+1)))
    a_data.append(n/attribute_max[i])
data.append(np.array(a_data))

#각 클래스로 분류될 확률 구하기
res = sess.run(y, feed_dict = {x: data})

#출력함수
def print_result(result):
    for f in result:
        for i in range(f.shape[0]):
            print("%d(으)로 분류될 확률 : %.2f%%"%(i, 100*f[i]))

print_result(res)


