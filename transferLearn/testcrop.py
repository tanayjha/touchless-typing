from PIL import Image
import numpy as np
import cv2
# import cPickle


def CropImage():
	name = "foo1.jpeg"
	im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	im_gray = cv2.bitwise_not(im_gray)
	ret, thresh = cv2.threshold(im_gray, 100,255,cv2.THRESH_BINARY)
	kernel = np.ones((3,3),np.uint8)
	im_gray = cv2.dilate(thresh,None,iterations = 5)
	im_gray = im_gray[100:400, 100:570]
	_, contours, hier = cv2.findContours(im_gray.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[-1]
	x,y,w,h = cv2.boundingRect(cnt)
	crop_img = im_gray[y:y+h, x:x+w]
	im_gray = cv2.bitwise_not(crop_img)
	im_gray = cv2.resize(im_gray, (28, 28))
	cv2.imwrite("foo1"+ ".jpeg", im_gray)
	# number = cPickle.load(open('save.p','rb'))
	# cv2.imwrite("d/fod" + str(number) + ".jpeg", im_gray)
	# number = number+1
	# cPickle.dump(number, open('save.p','wb'))