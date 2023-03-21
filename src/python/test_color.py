
import cv2
import numpy as np

path = "/home/albert/Downloads/Telegram Desktop/BBCPersian/photos/photo_8004@25-06-2019_15-42-56.jpg"

frame = cv2.imread(path)


# Converts images from BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110 ,50 ,50])
upper_blue = np.array([130 ,255 ,255])

lower_red = np.array([0 ,100 ,100])
upper_red = np.array([10 ,255 ,255])

lower_orange = np.array([10	,50 ,50])
upper_orange = np.array([20	,255 ,255])

lower_green = np.array([50	,50 ,50])
upper_green = np.array([100	,255 ,255])

# Here we are defining range of bluecolor in HSV
# This creates a mask of blue coloured
# objects found in the frame.
mask = cv2.inRange(hsv, lower_red, upper_red)

# The bitwise and of the frame and mask is done so
# that only the blue coloured objects are highlighted
# and stored in res
res = cv2.bitwise_and(frame ,frame, mask= mask)
cv2.imshow('frame' ,frame)
cv2.imshow('mask' ,mask)
cv2.imshow('res' ,res)
cv2.waitKey(0)


green = np.uint8([[[0, 255, 0]]]) #here insert the bgr values which you want to convert to hsv
hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print(hsvGreen)

lowerLimit = hsvGreen[0][0][0] - 10, 100, 100
upperLimit = hsvGreen[0][0][0] + 10, 255, 255

print(upperLimit)
print(lowerLimit)