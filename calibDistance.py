from djitellopy import Tello
import cv2
from time import sleep
from imutils import grab_contours
tello = Tello()
tello.connect()
tello.streamon()
sleep(2)
frame_read = tello.get_frame_read()
image = frame_read.frame
h, w = image.shape[:2]
print(h, w)

def main():
    while True:
        image = frame_read.frame
        rect = cv2.selectROI(image)
        print(rect)
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('test2', image)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        ret, img = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
        canny = cv2.Canny(img, 20, 100)
        cv2.imshow('test', canny)
        cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)
        c = cv2.minAreaRect(c)
        '''
        #print('w %d h %d cnt %s'%(w, h, str(c)))
        key = cv2.waitKey()
        if key == 27: break

if __name__ == '__main__':
    main()
    tello.end()