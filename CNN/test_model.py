import tensorflow as tf
import train_model
import dlib
import os
import sys
import cv2


input_dir = 'face_recog/test_faces'
index = 1
output = train_model.cnnLayer()
predict = tf.argmax(output, 1)
# 先加载meta graph并回复权重变量
saver = tf.train.import_meta_graph('.train_face_model/train_faces.model.meta')
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.train_face_model/'))

#user = input('图片(G)还是摄像头(V)：')
def is_my_face(image):
    sess.run(tf.global_variables_initializer())
    res = sess.run(predict, feed_dict={train_model.x: [image / 255.0], train_model.keep_prob_5:1.0, train_model.keep_prob_75: 1.0})
    if res[0] == 0:
        return True
    else:
        return False

detector = dlib.get_frontal_face_detector()


for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                print('Being Processed picture %s' % index)
                index += 1
                # 读取图片
                img_path = path + '/' + filename
                img = cv2.imread(img_path)
                # 将图片变为灰度图片
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)
                if not len(dets):
                    print('can not get face')
                    cv2.imshow('image', img)
                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        sys.exit(0)
                for i, d_test in enumerate(dets):
                    x1 = d_test.top() if d_test.top() > 0 else 0
                    y1 = d_test.bottom() if d_test.bottom() > 0 else 0
                    x2 = d_test.left() if d_test.left() > 0 else 0
                    y2 = d_test.right() if d_test.right() > 0 else 0
                    face = img[x1:y1, x2:y2]
                    # 调整图片格式
                    face = cv2.resize(face, (train_model.size, train_model.size))
                    print('Is my face? %s' % is_my_face(face))
                    cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
                    cv2.imshow('image', img)
                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        sys.exit(0)


sess.close()


