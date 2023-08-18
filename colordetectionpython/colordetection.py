import cv2
import numpy as np
import imutils


vid = cv2.VideoCapture(0)
vid.set(5, 640)
vid.set(4, 480)

while(True):

    
    sucess, img = vid.read()

    #img = cv2.flip(img, 1)
    #img = cv2.flip(img, 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([10, 180, 70]) # test with 0, 140, 140 # black is 0, 0, 0 # skin color is 240, 100, 100
    upper_bound = np.array([40, 255, 255])# test with 180, 255, 255 # black is 180, 255, 30
    lower_bound2 = np.array([150, 50, 50]) # red is 150, 50, 50
    upper_bound2 = np.array([180,255,255]) # red is 180, 255, 255

    masking1 = cv2.inRange(hsv, lower_bound, upper_bound)
    masking2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    masking = masking1 + masking2
    blurred = cv2.blur(masking, (2, 2))
    #ret, thresh = cv2.adaptiveThreshold(blurred, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        contours = max(contours, key=lambda x: cv2.contourArea(x))
    
        cv2.drawContours(img, [contours], -1, (255,255,0), 2)

        M = cv2.moments(contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
	
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(img, (cX - 100,cY - 100), (cX + 100, cY + 100), (0, 255, 0), 3)
        cv2.rectangle(img, (290, 450), (350,480), (0, 255, 0), 3)
        cv2.imshow('frame', img)
    except KeyError:
        cv2.rectangle(img, (290, 450), (350,480), (0, 255, 0), 3)
        cv2.imshow('frame', img)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        
        break

vid.release()
cv2.destroyAllWindows()
