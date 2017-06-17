from threading import Thread
import sys
# import the Queue class from Python 3
from queue import Queue
from imutils.video import FileVideoStream
import imutils
import cv2
import time
import numpy as np
from threading import Thread
 
class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
 
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
 
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
 
	def read(self):
		# return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2

vs = WebcamVideoStream(src=1).start()
 
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame, 1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()