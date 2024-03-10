# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread('./data/CCPD2020/ccpd_green/val/0255772569444-92_249-215&478_499&569-499&569_237&549_215&478_490&489-0_0_3_24_33_29_26_30-100-25.jpg')
source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 高斯滤波
result = cv2.GaussianBlur(source, (3, 3), 0)  # 可以更改核大小
# 显示图形
titles = ['Source Image', 'GaussianBlur Image (3, 3)']
images = [source, result]
cv2.imshow('image_raw', source)
cv2.imshow('image_gamma', result)
cv2.waitKey(0)

