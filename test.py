from torch.utils.data import Dataset
from fake_chs_lp.random_plate import Draw
from torch import nn
import os
from torchvision.transforms import transforms
from einops import rearrange
import random
import cv2
from utils_1 import enhance, make_label
import numpy as np
import torch
import ocr_config
import detect_config
import re


class OcrDataSet(Dataset):

    def __init__(self):
        super(OcrDataSet, self).__init__()
        self.dataset = []
        self.draw = Draw()
        for i in range(100000):
            self.dataset.append(1)
        self.smudge = enhance.Smudge()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        plate, label = self.draw()
        target = []
        for i in label:
            target.append(ocr_config.class_name.index(i))
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        '''数据增强'''
        plate = self.data_to_enhance(plate)

        # cv2.imshow('a', plate)
        #cv2.imwrite('1.jpg', plate)
        # cv2.waitKey()

        image = torch.from_numpy(plate).permute(2, 0, 1) / 255
        #image = torch.nn.functional.pad(image, (0, 15000, 0, 0), mode='constant', value=0)
        # image = self.transformer(image)
        # print(image.shape)
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        _target[:len(target)] = target
        return image, _target, target_length


    def data_to_enhance(self, plate):
        '''随机污损'''
        plate = self.smudge(plate)
        '''高斯模糊'''
        plate = enhance.gauss_blur(plate)
        '''高斯噪声'''
        plate = enhance.gauss_noise(plate)
        '''增广数据'''
        plate, pts = enhance.augment_sample(plate)
        '''抠出车牌'''
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate



def data_to_enhance(plate):
    '''随机污损'''
    plate = enhance.Smudge(plate)
    '''高斯模糊'''
    plate = enhance.gauss_blur(plate)
    '''高斯噪声'''
    plate = enhance.gauss_noise(plate)
    '''增广数据'''
    plate, pts = enhance.augment_sample(plate)
    '''抠出车牌'''
    plate = enhance.reconstruct_plates(plate, [np.array(pts).reshape((2, 4))])[0]
    return plate



def Get_Train_DataSet():
    """
    input your traindir to path
    this function will get dataset for trainloader
    """
    draw = Draw()
    name,plate = draw()
    dataset = []
    path = r'F:\License_Detection\data\CCPD2020\ccpd_green\lpr_train'

    for filename in os.listdir(path):

        file_path = os.path.join(path, filename)
        # plate = cv2.imread(file_path)
        plate = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        plate = cv2.resize(plate, (144, 48))
        # print(plate.shape)
        # plate = data_to_enhance(plate)
        image = torch.from_numpy(plate).permute(2, 0, 1) / 255


        label = filename.split('.')[0]
        target = []
        for i in label:
            target.append(ocr_config.class_name.index(i))

        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()

        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)

        _target[:len(target)] = target

        dataset.append((image, _target, target_length))

        # cv2.imshow('a', plate)
        # cv2.waitKey(100);
    return dataset


