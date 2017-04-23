# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.preprocessing import scale
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
from PIL import Image
import glob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
#print("[INFO] downloading MNIST...")
#dataset = datasets.fetch_mldata("MNIST Original")

image_list_0 = np.empty((0))
for filename in glob.glob('Data/TrainingData/0/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0	
	image_list_0=np.append(image_list_0,arr)

image_list_0=image_list_0.reshape(100,1,150,200).astype('float32')

image_list_1 = np.empty((0))
for filename in glob.glob('Data/TrainingData/1/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_list_1=np.append(image_list_1,arr)

image_list_1=image_list_1.reshape(100,1,150,200).astype('float32')
 
image_list_2 = np.empty((0))
for filename in glob.glob('Data/TrainingData/2/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_list_2=np.append(image_list_2,arr)

image_list_2=image_list_2.reshape(100,1,150,200).astype('float32')

image_list_3 = np.empty((0))
for filename in glob.glob('Data/TrainingData/3/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_list_3=np.append(image_list_3,arr)

image_list_3=image_list_3.reshape(100,1,150,200).astype('float32')

image_list_4 = np.empty((0))
for filename in glob.glob('Data/TrainingData/4/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_list_4=np.append(image_list_4,arr)

image_list_4=image_list_4.reshape(100,1,150,200).astype('float32')

image_list_5 = np.empty((0))
for filename in glob.glob('Data/TrainingData/5/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_list_5=np.append(image_list_5,arr)

image_list_5=image_list_5.reshape(100,1,150,200).astype('float32')


image_list=np.concatenate((image_list_0,image_list_1,image_list_2,image_list_3,image_list_4,image_list_5),axis = 0)

#print image_list.shape

image_labels = np.empty((0))

for i in range(100):
	image_labels=np.append(image_labels,0)

for i in range(100):
	image_labels=np.append(image_labels,1)

for i in range(100):
	image_labels=np.append(image_labels,2)

for i in range(100):
	image_labels=np.append(image_labels,3)

for i in range(100):
	image_labels=np.append(image_labels,4)

for i in range(100):
	image_labels=np.append(image_labels,5)

image_labels=np_utils.to_categorical(image_labels,6)

image_test = np.empty((0))

image_test_0 = np.empty((0))
for filename in glob.glob('Data/TestData/0/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0	
	image_test_0=np.append(image_test_0,arr)

image_test_0=image_test_0.reshape(20,1,150,200).astype('float32')

image_test_1 = np.empty((0))
for filename in glob.glob('Data/TestData/1/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_test_1=np.append(image_test_1,arr)

image_test_1=image_test_1.reshape(20,1,150,200).astype('float32')
 
image_test_2 = np.empty((0))
for filename in glob.glob('Data/TestData/2/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_test_2=np.append(image_test_2,arr)

image_test_2=image_test_2.reshape(20,1,150,200).astype('float32')

image_test_3 = np.empty((0))
for filename in glob.glob('Data/TestData/3/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_test_3=np.append(image_test_3,arr)

image_test_3=image_test_3.reshape(20,1,150,200).astype('float32')

image_test_4 = np.empty((0))
for filename in glob.glob('Data/TestData/4/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_test_4=np.append(image_test_4,arr)

image_test_4=image_test_4.reshape(20,1,150,200).astype('float32')

image_test_5 = np.empty((0))
for filename in glob.glob('Data/TestData/5/*.jpeg'):
	im=Image.open(filename,'r').convert('L')
	arr=np.asarray(im.getdata(),dtype=np.float64).reshape((im.size[1],im.size[0]))
	arr=arr/255.0
	image_test_5=np.append(image_test_5,arr)

image_test_5=image_test_5.reshape(20,1,150,200).astype('float32')


image_test=np.concatenate((image_test_0,image_test_1,image_test_2,image_test_3,image_test_4,image_test_5),axis =0)

label_test = np.empty((0))


for i in range(20):
	label_test=np.append(label_test,0)

for i in range(20):
	label_test=np.append(label_test,1)

for i in range(20):
	label_test=np.append(label_test,2)

for i in range(20):
	label_test=np.append(label_test,3)

for i in range(20):
	label_test=np.append(label_test,4)

for i in range(20):
	label_test=np.append(label_test,5)


label_test=np_utils.to_categorical(label_test,6)

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits

#data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
#data = data[:, np.newaxis, :, :]
#(trainData, testData, trainLabels, testLabels) = train_test_split(
#	data / 255.0, dataset.target.astype("int"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
#trainLabels = np_utils.to_categorical(trainLabels, 10)
#testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
#print image_list.shape
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = LeNet.build(width=200, height=150, depth=1, classes=6,
	weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(image_list, image_labels, batch_size=20, nb_epoch=10,
		verbose=1)

	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	for i in range(120):
		print i
		probs = model.predict(image_test[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		print prediction

	(loss, accuracy) = model.evaluate(image_test, label_test,
		batch_size=4, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
#for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
	# classify the digit
#	probs = model.predict(testData[np.newaxis, i])
#	prediction = probs.argmax(axis=1)

	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
#	image = (image_test[i][0] * 255).astype("uint8")
#	image = cv2.merge([image] * 3)
#	image = cv2.resize(image, (150,200), interpolation=cv2.INTER_LINEAR)
#	cv2.putText(image, str(prediction[0]), (5, 20),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
#	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
#		np.argmax(testLabels[i])))
#	cv2.imshow("Digit", image)
	cv2.waitKey(0)