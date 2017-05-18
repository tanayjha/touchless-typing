# import numpy as np
# import cv2
# import cPickle
# from collections import deque

from collections import deque
import numpy as np
import cv2
import matplotlib
from plot import plotting

camera = cv2.VideoCapture(1)
ret, frame = camera.read()

if not camera.isOpened():
	print ("video not opened")
	exit()

itemlist = []
pts = deque()
counter = 40
while (True):
  	ret, frame = camera.read()
  	frame = cv2.flip(frame, 1)
  	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  	ret, thresh = cv2.threshold(gray, 250,255,cv2.THRESH_BINARY)
	thresh = cv2.dilate(thresh, None, iterations=7)
	kernel = np.ones((5,5),np.float32)/25
	gray = cv2.filter2D(thresh,-1,kernel)
	thresh = gray
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		counter = 0
		cnt = contours[0]
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
		pts.appendleft([int(x+w/2),int(y+h/2)])
	else:
		counter = (counter+1) % 100000
		if counter==5:
			plotting(list(pts))
			pts.clear()
  	cv2.imshow('frame',frame)
  	if cv2.waitKey(1) & 0xFF == ord('q'):
  		break

camera.release()
cv2.destroyAllWindows()