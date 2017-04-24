from PIL import Image
import numpy
import os

def reduceImage(filename):
	fName =  os.path.splitext(os.path.basename(filename))[0]
	ext = os.path.splitext(os.path.basename(filename))[-1]
	im = Image.open(filename).convert("L")

	(width, height) = im.size
	greyscale_map = list(im.getdata())
	greyscale_map = numpy.array(greyscale_map)
	greyscale_map = greyscale_map.reshape((height, width))

	colSum = greyscale_map.sum(axis = 0)
	rowSum = greyscale_map.sum(axis = 1)

	maxColSum = height*255
	maxRowSum = width*255

	for left in range(width):
		if colSum[left] < maxColSum:
			break

	for right in range(width-1,-1,-1):
		if colSum[right] < maxColSum:
			break

	for top in range(height):
		if rowSum[top] < maxRowSum:
			break
	for bottom in range(height-1,-1,-1):
		if rowSum[bottom] < maxRowSum:
			break

	left  = max(0,left-2)
	right = min(right+2,width-1)
	top = max(0,top-2)
	bottom = min(bottom+2,height-1)


	if (right-left) > (bottom-top):
		top = max(0,top-(((right-left) - (bottom-top))/2))
		bottom = min(height-1,bottom+((right-left) - (bottom-top))/2)

	else:
		left = max(0,left-((bottom-top)-(right-left))/2)
		right = min(width-1,right+((bottom-top)-(right-left))/2)

	im1 = im.crop((left,top,right,bottom)).resize((100,100), Image.ANTIALIAS)
	im1.save(fName+"Reduced"+ext)