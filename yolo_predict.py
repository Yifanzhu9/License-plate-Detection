import cv2
import os
from ultralytics import YOLO

folder_path = r'F:\License_Detection\data\CCPD2020\ccpd_green\det\images\test\\'
filenames= os.listdir(folder_path)

model= YOLO('F:\License_Detection\weights\yolov5su.pt')


for filename in filenames:
    filepath = os.path.join(folder_path,filename)
    img = cv2.imread(filepath)
    result = model.predict(source=img, save=True, name=r'F:\License_Detection\data\yolo_prediction' + filename)
