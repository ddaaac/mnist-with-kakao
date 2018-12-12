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
                   자세한 건 첨부한 StudentsPerformance.csv 참고

'''

attri_num = 3  #사용할 속성 수 입력 - 읽어올 수도 있는데 이렇게 직접 쓰면 좀 더 이해가 되지 않을까싶어서 직접 입력하도록 함.
class_num = 21   #사용할 분류 수 입력 - 마찬가지


train_data = []
label_data = []

read_file = pd.read_csv('./BlackFriday.csv')
label_read = read_file[['label']].values
read_file = read_file.drop('label',1)

attribute_max = []
maxidx = read_file.abs().idxmax()
#해당 속성이 갖는 최대값으로 나누기 -> 0~1사이로 보정
for idx, col in enumerate(read_file):
    mean = read_file[col].values.mean()
    std = read_file[col].values.std()
    an_attribute_max = float(read_file.loc[maxidx[idx],col]) 
    if (an_attribute_max != 0):
        read_file[col] = read_file[col] / an_attribute_max
        #read_file[col] = (read_file[col]-mean)/std
        print(read_file[col])
    attribute_max.append(an_attribute_max) #속성별 최댓값 저장 -> 나중에 test case 입력 받고 그 값들도 나눠줘야되니깐.
print(attribute_max)
'''
#test case의 최대값과 실제 input의 최댓값이 다른 경우
#예를들어, 테스트 케이스에는 0~95의 값밖에 없는데 실제로는 100이 최대라면 속성별 값의 범위를 다시 설정해줘야함
if_should_modify_max_attribute = input("각 속성별 최대값이 테스트 케이스와 다른경우가 있나요?(y/n)")

if(if_should_modify_max_attribute == 'y'):
    for i in range(attri_num):
        n = input("{}번째 속성의 최댓값 입력(넘어가려면 enter) : ".format(i+1))
        if(n):
            attribute_max[i] = float(n)
'''
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
    if(idx < 3):
        print(sess.run(W))

    '''
    #테스트용 출력
    if(idx > 0 and idx < 10):
        print(idx)
        print(sess.run(cross_entropy,feed_dict={x:batch_xs, y_ : batch_ys}))
        print(sess.run(W))
        print(sess.run(y,feed_dict={x:batch_xs}))
    '''

#save model
#saver = tf.train.Saver()
#save_path = saver.save(sess,'C:\\Users\\Kim BeomJun\\Desktop\\kim\\mnist\\minist_softmax.ckpt' )
#saver.restore(sess, 'C:\\Users\\Kim BeomJun\\Desktop\\kim\\mnist\\minist_softmax.ckpt')

#test data 입력
data = []
a_data = []
for i in range(attri_num):
    '''
    while True:
        n = int(input("{}번째 속성 값 입력 : ".format(i+1)))
        if(n > attribute_max[i]):
            print("다시 입력해주세요")
        else:
            break
    '''
    n = float(input("{}번째 속성 값 입력 : ".format(i+1)))
    print(n)
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


