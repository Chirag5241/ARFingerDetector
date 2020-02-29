import cv2
import sys
import numpy as np


# img = cv2.imread(sys.argv[1])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.GaussianBlur(img, (7,7), 0)
#
# red_lower_bound = np.array([0,60,80])
# red_upper_bound = np.array([10,256,256])
#
# red_mask = cv2.inRange(img, red_lower_bound, red_upper_bound)
# masked_img = cv2.bitwise_and(img, img, mask = red_mask)
# masked_img = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)
#
# cv2.imshow('masked_img', masked_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cap = cv2.VideoCapture('movingcars.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

red_lower_bound = np.array([0,60,80])
red_upper_bound = np.array([10,256,256])


if(cap.isOpened() == False):
    print('Error opening video stream or file')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        fgmask = fgbg.apply(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('Frame', fgmask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.realease()
cv2.destroyAllWindows()
