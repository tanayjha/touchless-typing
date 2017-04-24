from PIL import Image
import numpy as np
import cv2
for i in range(0, 1):
	name = "foo" + str(i+1) + ".jpeg"
	im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	im_gray = im_gray[100:440, 120:700]	
	(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
	im_bw = cv2.bitwise_not(im_bw)
	kernel = np.ones((3,3),np.uint8)
	im_bw = cv2.dilate(im_bw,kernel,iterations = 5)	
	contours, hier = cv2.findContours(im_bw.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[-1]
	draw = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(draw,[cnt],0,(0,255,0), 2)
	x,y,w,h = cv2.boundingRect(cnt)
	crop_img = draw[y:y+h, x:x+w]
	cv2.rectangle(draw,(x,y),(x+w,y+h),(0,0,255),2)
	imgray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	resized_image = cv2.resize(thresh, (28, 28))
	final_name = "final_u" + str(i+1) + ".jpeg"
	cv2.imwrite(final_name, resized_image)
	cv2.imshow('image', resized_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	