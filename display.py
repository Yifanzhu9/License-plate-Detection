from kenshutsu import Get_Picture_Information
import cv2



image = cv2.imread('test_picture.jpg')
result = Get_Picture_Information(image)
image = result[0]

image = cv2.resize(image, (640, 640))
cv2.imshow('a', image)
cv2.waitKey()

print(result[1])