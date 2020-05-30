''' 导入需要的库 '''
import tensorflow as tf
import cv2
import numpy as np
import os #可以将训练的模型保存的库
import random
import sys
from sklearn.model_selection import train_test_split

""" 第一步：定义预处理后的图片所在目录 """
my_faces_path = 'my_faces'
other_faces_path = 'other_faces'
size = 64

""" 第二步：调整图片的大小 """
imgs = []
labs = []
# 设置默认的数据流图
tf.reset_default_graph()
# 获取需要填充图片的大小
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longset = max(h, w)
    if w < longset:
        temp = longset - w
        # //表示整除
        left = temp // 2
        right = temp - left
    elif h < longset:
        temp = longset - h
        top = temp // 2
        bottom = temp - top
    else:
        pass
    return top, bottom, left, right

""" 第三步：读取测试图片 """
def readTestPicture(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)
readTestPicture(my_faces_path)
readTestPicture(other_faces_path)
# 将图片数据与标签转换成数据
imgs = np.array(imgs)
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs]) # 前面112个位[0， 1],后面的是[1, 0]
# 随机划分训练集和测试集 random.randint(0, 100)：产生0-100的整数。95%用来训练，5%用来测试
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
# 参数：图片数据的总数，图片的高、宽、颜色通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 图片数据的归一化
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
print('train_size:%s, test_size:%s' % (len(train_x), len(test_x)))
# 图片块， 每次取100张图片
batch_size = 20
num_batch = len(train_x) // batch_size

""" 第四步：定义变量及神经网络 """
# 占位符定义
# 参数：图片数据的类型、图片数据的总数、图片的高、宽、颜色通道
x = tf.placeholder(tf.float32, [None, size, size, 3])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# 网络权重和偏置的定义
'''
.Variable(): 该构造方法创建Variable对象
Variable对象初值通常为全0、全1或者用随机数填充阶数较高的张量
创建初值的Op有：.zeros()、 ones()、 random_normal()、 random_uniform(),接收shape参数
'''
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01) # 创建初值Op，接收两个输入，服从方差为0.01的正态分布初值。sess(Op)：必须将初始化Op传给Session.run()，实现Varaible对象在Session对象内初始化
    return tf.Variable(init) # 创建Variable对象
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)
# 卷积层定义
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1 ,1], padding='SAME')
# 池化层定义
'''
tf.nn.max_pool(value, ksize, strides, padding)
value:池化的输入，输入的类型是[batch, height, width, channels]的shape
ksize:池化窗口的大小，一般是[1, height, width, 1]
strides:和卷积类似，窗口在每一个维度上的滑动步长
'''
def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
#对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络
def dropout(x, keep):
    return tf.nn.dropout(x, keep)


""" 第五步： 定义卷积神经网络框架 """
def cnnLayer():
    ''' 第一层'''
    w1 = weightVariable([3, 3, 3, 32]) #权重，卷积核的大小(3, 3)，输入颜色通道(3)，输出通道大小(32)
    b1 = biasVariable([32])
    # 1.卷积 要经过激活函数再输出
    conv1 = tf.nn.relu(conv2d(x, w1) + b1)
    # 2.池化
    pool1 = maxpool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    ''' 第二层'''
    # 权重和偏置
    w2 = weightVariable([3, 3, 32, 64]) # 卷积核代大小(3, 3),输入通道(32)是上一层的输入, 输出通道(64)
    b2 = biasVariable([64])
    # 1.卷积层
    conv2 = tf.nn.relu(conv2d(drop1, w2) + b2)
    # 2.池化层
    pool2 = maxpool(conv2)
    #  减少过拟合，随机让某些权重不更新
    drop2 = dropout(pool2, keep_prob_5)

    """ 第三层 """
    # 权重和偏置
    w3 = weightVariable([3, 3, 64, 64]) # 卷积核代大小(3, 3),输入通道(32)是上一层的输入, 输出通道(64)
    b3 = biasVariable([64])
    # 1.卷积
    conv3 = tf.nn.relu(conv2d(drop2, w3) + b3)
    # 2.池化
    pool3 = maxpool(conv3)
    # 3.减少过拟合，随机让某些权重不更新
    drop3 = dropout(pool3, keep_prob_5)

    """ 全连接层 """
    #权重和偏置
    wf = weightVariable([8 * 16 * 32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    """ 输出层 """
    w_out = weightVariable([512, 2])
    b_out = biasVariable([2])
    out = tf.add(tf.matmul(dropf, w_out), b_out) # 定义加法Op， 接收两个张量，输出张量的和
    return out

""" 第六步：训练模型 """
def cnnTrain_model():
    out = cnnLayer()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    train_op = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等， 再求所有书的平均值，tf.cast(强制转换类型) argmax(out, 1):求每一行的最大值
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), tf.float32))
    # 将loss和accuracy保存以便tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy )
    # summary融合
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器初始化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # tensorboard中变量初始化
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./face_recog',graph=tf.get_default_graph())
        for n in range(10):
            # 每次取128(batch_size)张图片
           for i in range(num_batch):
                batch_x = train_x[i * batch_size : (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _, loss, summary = sess.run([train_op, cross_entropy, merged_summary_op],
                                                    feed_dict={x:batch_x, y: batch_y, keep_prob_5:0.5, keep_prob_75:0.75})
                summary_writer.add_summary(summary, n * num_batch + i)
                # 输出损失
                print(n * num_batch + i, loss)
                if (n * num_batch + i) % 40 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x: test_x, y: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print(n * num_batch + i, acc)
                    if acc > 0.8 and n > 2:
                        saver.save(sess, '.train_face_model/train_faces.model')


# cnnTrain_model()