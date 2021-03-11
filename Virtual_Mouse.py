import urllib.request
import ssl

import imutils
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

from pynput.mouse import Button, Controller

mouse = Controller()

cap = cv2.VideoCapture(0)
pre = [0,0,0,0]
cur = [0,0,0,0]
d=[0,0]
flag = 0
count = 0
occ =0
fl =0
def find(img,image):
    #image = img
    
    global cur,pre,flag,count,occ,fl
    occ+=1
    #imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(grey_image, (11, 11), 0)

    threshold = 0

    thresh = cv2.threshold(blurred,threshold,255,cv2.THRESH_BINARY)[1]

    #cv2.imshow('Threshold Applied',thresh)

    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)

    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):

            if label == 0:
                    continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255

            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 300:
                    mask = cv2.add(mask, labelMask)
                    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts)[0]
        
    for (i, c) in enumerate(cnts):
            
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (int(x), int(y)),(int(x+w),int(y+h)),(0, 0, 255), 3)
            cv2.putText(image, "#POINTER", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            #print((x,y),(x+w,y+h))
            cur = [x,y,w,h]
    diff = -cur[2]*cur[3] + pre[2]*pre[3]
    
    if abs(diff)>5:
        x = cur[0] - pre[0]
        y = pre[1]-cur[1]  #-cur[2]*cur[3] + pre[2]*pre[3]
        if cur[1]>430 and fl==2:
            fl=0
            mouse.release(Button.left)
        if abs(y)>10 and cur[1]<440 and fl!=2:
            y=0
        if cur[1] < 420 and fl!=2:
            pre = cur
            fl =2
            mouse.press(Button.left)
            flag += 1
            if flag==1:
                count=1
                print('Left Click')
            else:
                print('Double Click')
                
        elif cur[1] < 420 and fl!=2:
            fl =1
            print('Right Click')
            mouse.press(Button.right)
            mouse.release(Button.right)
            
        else:
            print(cur,x,y)
            mouse.move(3*x,5*y)
            pre=[cur[0],cur[1],cur[2],cur[3]]
            
                
    if count>=1:
        count+=1
        if count<=5 and flag==2:
            mouse.click(Button.left, 2)
            count = 0
            flag = 0
        elif count >5:
            count = 0
            flag = 0
    cv2.imshow('frame',image)
    
def skin(image):

    lower = np.array([0, 48, 0], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    frame = image
    #frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    # show the skin in the image along with the mask
    find(skin,image)
  
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#url = 'http://192.168.43.1:8080/shot.jpg'  #j7p
url = 'http://192.168.43.219:8080/shot.jpg' #a51
while(True):

    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    #img = cv2.imread(r'C:/Users/ranji/Desktop/pointer.jpg')
    skin(img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
