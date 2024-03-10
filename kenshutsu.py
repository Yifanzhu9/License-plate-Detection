# import argparse
import os
import sys
from pathlib import Path
import numpy
import torch
# import torch.backends.cudnn as cudnn
from read_plate import ReadPlate


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

# from utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
from PIL import Image, ImageDraw, ImageFont

import matplotlib.font_manager as fm # to create font
from PIL import Image,ImageFont,ImageDraw






class Kenshutsu(object):

    def __init__(self, is_cuda = True):

        device = 'cuda:0' if is_cuda and torch.cuda.is_available() else 'cpu'
        # the weights of yolov5 model
        weights = r'F:\License_Detection\yolov5\runs\train\exp19\weights\best.pt'

        if not os.path.exists(weights):
            raise RuntimeError('Model parameters not found')

        self.device = select_device(device)

        # load yolo model
        self.model = DetectMultiBackend(weights, device=self.device)
        imgsz = (640, 640)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)
        bs = 1 # batch_size

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.agnostic_nms = False
        self.classes = None
        self.iou_thres = 0.45
        self.conf_thres = 0.25


    def __call__(self, image):
        h, w, c = image.shape
        image, h2, w2, fx = self.square_picture(image, 640)
        image_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = numpy.transpose(image_tensor, axes=(2, 0, 1)) / 255
        image_tensor = torch.from_numpy(image_tensor).float().to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        pred = self.model(image_tensor)
        pred = \
        non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=1000)[0]
        boxes = pred.cpu()
        result = []
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = max(0, int((x1 - (640 // 2 - w2 // 2)) / fx)), max(0, int((y1 - (
                        640 // 2 - h2 // 2)) / fx)), min(w, int((x2 - (640 // 2 - w2 // 2)) / fx)), min(h, int((y2 - (
                        640 // 2 - h2 // 2)) / fx))
            result.append([x1, y1, x2, y2, the, c])
        return result

    @staticmethod
    def square_picture(image, image_size):
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        if max_len >= image_size:
            fx = image_size / max_len
            fy = image_size / max_len
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 127
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background, h2, w2, fx
        else:
            h2, w2, _ = image.shape
            background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
            background[:, :, :] = 127
            s_h = image_size // 2 - h2 // 2
            s_w = image_size // 2 - w2 // 2
            background[s_h:s_h + h2, s_w:s_w + w2] = image
            return background, h2, w2, 1



def DrawChinese(img, text, positive, fontSize=100, fontColor=(255, 0, 0)):  # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    #font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    font = ImageFont.truetype('simsun.ttc', fontSize, encoding="utf-8")
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    cv2charimg = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return cv2charimg



class Get_Information(object):

    def __init__(self):
        self.detecter = Kenshutsu()
        self.read_plate = ReadPlate()

    def __call__(self, image):
        boxes = self.detecter(image)
        plates = []

        plate_names = []
        thes = [] # confidence
        if len(boxes) == 0:
            # print("There is no plate now")
            # print(image.shape)
            return (image, [], [], [])

        else :
            for box in boxes:
                x1, y1, x2, y2, the, c = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                the = float(the)
                thes.append(the)
                # print(x1, y1, x2, y2, the, c)
                # if c == 2 or c == 5:
                # input the picture of license plate -- image_
                image_ = image[y1:y2, x1:x2]
                license_box = [x1, y1, x2, y2]
                result = self.read_plate(image_, license_box)

                # if result:
                plate, (x11, y11, x22, y22) = result[0]

                # plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
                plates.append((x1, y1, x2, y2, plate))

            for plate in plates:
                x1, y1, x2, y2, plate_name = plate
                plate_names.append(plate_name)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
                image = DrawChinese(image, plate_name, (x1, y2))
            # image = cv2.resize(image, None, fx=0.5, fy=0.5)
            # image image_name thes
            colors = []


            for License in plate_names:
                if len(License) == 8:
                    colors.append('Green')
                elif len(License) == 7:
                    colors.append('Blue')
                else:
                    colors.append('Unknown license plate')
            print("The recognized plates are as follows: ")
            print(plate_names)
            print("Get Successfully!")
            return (image, plate_names, thes, colors)





def Get_Picture_Information(image):
    """
    :return:
    image after handle
    content of license plate
    yolov5 confidence
    """
    detecter = Kenshutsu(False)
    read_plate = ReadPlate()
    boxes = detecter(image)
    plates = []

    plate_names = []
    thes = [] # confidence
    if len(boxes) == 0:
        return (image, [], [], [])

    else :
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            the = float(the)
            thes.append(the)
            # print(x1, y1, x2, y2, the, c)
            # if c == 2 or c == 5:
            # input the picture of license plate -- image_
            image_ = image[y1:y2, x1:x2]
            license_box = [x1, y1, x2, y2]
            result = read_plate(image_, license_box)

            # if result:
            plate, (x11, y11, x22, y22) = result[0]

            # plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
            plates.append((x1, y1, x2, y2, plate))

        for plate in plates:
            x1, y1, x2, y2, plate_name = plate
            plate_names.append(plate_name)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
            image = DrawChinese(image, plate_name, (x1, y2))
        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        # image image_name thes
        colors = []


        for License in plate_names:
            if len(License) == 8:
                colors.append('Green')
            elif len(License) == 7:
                colors.append('Blue')
            else:
                colors.append('Unknown license plate')
        print("The recognized plates are as follows: ")
        print(plate_names)
        print("Get Successfully!")
        return (image, plate_names, thes, colors)



if __name__ == '__main__':
    import os
    class_name = ['main']

    # dir for pictures waiting for being drawn
    root = r'F:\License_Detection\data\CCPD2020\ccpd_green\det\images\test'
    detecter = Kenshutsu(False)

    read_plate = ReadPlate()
    count = 0

    # image = cv2.imread('test1.jpg')
    # boxes = detecter(image)
    # for box in boxes:
    #     x1, x2, x3,x4,the,c = box
    #     print(c)


    for image_name in os.listdir(root):
        image_path = f'{root}/{image_name}'
        image = cv2.imread(image_path)
        boxes = detecter(image)
        plates = []
        for box in boxes:
            x1, y1, x2, y2, the, c = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2, the, c)
            # if c == 2 or c == 5:
                # input the picture of license plate -- image_
            image_ = image[y1:y2, x1:x2]
            license_box = [x1, y1, x2, y2]
            result = read_plate(image_, license_box)

                #if result:
            plate, (x11, y11, x22, y22) = result[0]
            print(plate)
                # plates.append((x1, y1, x2, y2, plate, x11 + x1, y11 + y1, x22 + x1, y22 + y1))
            plates.append((x1, y1, x2, y2, plate))


        for plate in plates:
            x1, y1, x2, y2, plate_name = plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x11, y11, x22, y22 = int(x11), int(y11), int(x22), int(y22)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # image = cv2.rectangle(image, (x11 - 5, y11 - 5), (x22 + 5, y22 + 5), (0, 0, 255), 2)
            image = DrawChinese(image, plate_name, (x1, y2))

        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        print(image_name)
        cv2.imshow('a', image)
        cv2.waitKey()







