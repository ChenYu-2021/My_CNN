''' 导入模块 '''
import sys # 系统退出
import os # 创建文件夹，寻找文件的库
import cv2 # opencv
import matplotlib.pyplot as plt
import dlib # 人脸识别算法库
import time # 记录程序的库

start_time = time.time() # 程序开始时间

""" 第一步：获取数据 """
# 1.定义输入、输出目录
input_dir_myself = 'face_recog/my_faces'
out_dir_myself = 'my_faces'
size = 64

# 2.判断输出目录是否存在，不存在，则创建目录
if not os.path.exists(out_dir_myself):
    os.makedirs(out_dir_myself)

# 3.利用dlib的人脸特征提取器，使用dlib自带的frontal_face_detector作为我们的人脸特征提取器
detector = dlib.get_frontal_face_detector()

"""  第二步：预处理数据 """
''' 使用dlib来批量识别图片中的人脸部分，并对原图像进行预处理，并保存到指定目录下 '''
# 1. 预处理自己的图像

index = 1
for (path, dirnames, filenames) in os.walk(input_dir_myself):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being Processed picture %s' % index)
            img_path = path + '/' + filename
            # 从文件读取图片
            img = cv2.imread(img_path)
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测，dets为返回的结果
            dets = detector(gray_img, 1)
            # 使用使用 enumerate 函数遍历序列中的元素以及它们的下标
            '''
            #下标 i 即为人脸序号
            #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
            #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
            '''
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                # 调整图片的尺寸
                face = cv2.resize(face, (size, size))
                # 显示图片
                cv2.imshow('image', face)
                # 保存图片
                cv2.imwrite(out_dir_myself + '/' + str(index) + '.jpg', face)
                index += 1
                # 不断刷新图像,频率时间为30ms
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)


# 2.用同样的方式来预处理别人的头像
''' 定义输入、输出目录，文件放到当前目录other_faces下 '''
input_dir_other = 'face_recog/other_faces'
out_dir_other = 'other_faces'

# 2.1 判断输出目录是否存在，若不存在，则创建文件
if not os.path.exists(out_dir_other):
    os.makedirs(out_dir_other)

# 2.2 利用dlib自带的get_frontal_face_detector作为特征提取器
detector_other = dlib.get_frontal_face_detector()

# 2.3 别人图像的预处理
index_other = 1
for (path, dirnames, filenames) in os.walk(input_dir_other): #path是所有的文件夹的路径
    for filename in filenames:
        # 判断图像是否是jpg格式
        if filename.endswith('.jpg'):
            img_path = path + '/' + filename
            # 图片读取
            img = cv2.imread(img_path)
            # 将图片转为灰度图像
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用detector进行人脸检测，dets为返回的结果
            dets_other = detector(gray_img, 1)
            '''  使用enumerate函数遍历序列中的元素和序列 '''
            for i_other, d_other in enumerate(dets_other):
                x1 = d_other.top() if d_other.top() > 0 else 0
                y1 = d_other.bottom() if d_other.bottom() > 0 else 0
                x2 = d_other.left() if d_other.left() > 0 else 0
                y2 = d_other.right() if d_other.right() > 0 else 0

                face = img[x1:y1, x2:y2]
                # 调整图片尺寸
                face = cv2.resize(face, (size, size))
                # 显示图片
                '''
                imshow(winname, mat)
                     @param winname Name of the window.
                     @param mat Image to be shown
                '''
                cv2.imshow('image', face)
                # 保存图片
                cv2.imwrite(out_dir_other + '/' + str(index_other) + '.jpg', face)
                index_other += 1
                # 刷新图片的频率为30ms
                key_other = cv2.waitKey(30) & 0xff
                if key_other == 27:
                    sys.exit(0)
