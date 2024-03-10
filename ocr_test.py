# from fake_chs_lp.random_plate import Draw
# # from models.ocr_net2 import OcrNet
# from ocr_explorer import Explorer
# import cv2
# import torch
#
# draw = Draw()
# explorer = Explorer()
# yes = 0
# count = 0
# for i in range(1000):
#     plate, label = draw()
#     plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
#     plate = cv2.resize(plate, (144, 48))
#     cv2.imshow('a', plate)
#     a = explorer(plate)
#     if a == label:
#         yes += 1
#     count += 1
#     print(a)
#     # print(a)
#     # cv2.waitKey(0)
#
# print(yes / count, yes, count)
# # cv2.waitKey()


from fake_chs_lp.random_plate import Draw
# from models.ocr_net2 import OcrNet
from ocr_explorer import Explorer
import numpy as np
import torch
import cv2
import os

explorer = Explorer()
yes = 0
count = 0

folder_path = r'F:\License_Detection\data\CCPD2020\ccpd_green\lpr_val'

for filename in os.listdir(folder_path):

    file_path = os.path.join(folder_path, filename)
    # plate = cv2.imread(file_path)

    plate = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
    plate = cv2.resize(plate, (144, 48))
    cv2.imshow('a', plate)
    cv2.waitKey(100)

    label = filename.split('.')[0]
    a = explorer(plate)
    if a == label:
        yes += 1
    count += 1
    # print(a)
    # print(label)
    # print(a)
    # cv2.waitKey(0)
print(yes / count, yes, count)
# cv2.waitKey()