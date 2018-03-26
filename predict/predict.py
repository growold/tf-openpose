# -*- coding: utf-8 -*-


import sys
import os
sys.path.append('/data1/research/matt/docker/tf_pose_estimation/tf-openpose/src/')

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

import argparse
import logging
import time
import ast

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
import matplotlib.pyplot as plt
%matplotlib inline 
from scipy import misc

def get_keypoint(image,humans):
    '''
    输入：
    image，矩阵格式
    humans，是关键点信息，预测出来的结果
    
    输出：
    centers，关键点信息，格式：
    点信息，（x,y）,概率
    {1: [(142, 303), 10.328406150326954],
     2: [(154, 303), 6.621983647346497],
     3: [(154, 323), 7.118330538272858]}
    
    '''
    image_h, image_w = image.shape[:2]
    centers = {}
    for n,human in enumerate(humans):
        center_tmp = {}
        for i in range(common.CocoPart.Background.value):  # range(common.CocoPart.Background.value) = range(18)
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
    #        print(human.body_parts[i])
    #         human.body_parts[i].score
    #         human.body_parts[i].x
    #         human.body_parts[i].y
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            center_tmp[i] = [center,human.body_parts[i].score]
        centers[n] = center_tmp
    #        print(center)
    return centers

def PoseEstimatorPredict(image_path,plot = False,resolution ='432x368', scales = '[None]',model = 'mobilenet_thin'):
    '''
    input:
        image_path,图片路径，jpg
        plot = False,是否画图，如果True,两样内容，关键点信息+标点图片matrix
        resolution ='432x368', 规格
        scales = '[None]',
        model = 'mobilenet_thin'，模型选择
    
    output:
        plot为false，返回一个内容：关键点信息
        plot为true，返回两个内容：关键点信息+标点图片matrix
    '''
    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    image = common.read_imgfile(image_path, None, None)
    t = time.time()
    humans = e.inference(image, scales=scales)  # 主要的预测函数
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (image_path, elapsed))
    centers = get_keypoint(image,humans)                                # 拿上关键点信息

    if plot:
        # 画图的情况下：
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)  # 画图函数
        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Result')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return centers,image
    else:
        # 不画图的情况下：
        return centers

if __name__ == '__main__':
    
    # logger记录
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 预测
    image_path = '/data1/research/matt/docker/tf_pose_estimation/tf-openpose/images/hand1.jpg'
    %time centers,images_p = PoseEstimatorPredict(image_path,plot = True,model = 'mobilenet_thin')
    '''
        # 目前支持两种：mobilenet_thin以及cmu
        # model,还有那些model，一共六种，可见文档：/src/network.py
        # 下载方式：
        # $ cd models/graph/cmu
        # $ bash download.sh
    '''
    
    # 保存
    misc.imsave('/data1/research/matt/docker/tf_pose_estimation/tf-openpose/images/000000000569_1.jpg', images_p)
    
    
