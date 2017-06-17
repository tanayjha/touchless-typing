from collections import deque
import numpy as np
import cv2
import matplotlib
from plot import plotting
import tensorflow as tf
import numpy as np
import os
import cv2
# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache
import prettytensor as pt
import time
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data
from imutils.video import WebcamVideoStream
import imutils
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

vs = WebcamVideoStream(src=1).start()

images_train = []
cls_train = []
labels_train = []

for dig in range(0, 1):
    for imgno in range(0, 50):
        name = "ShivamTestdata/" + str(dig) + "/fo" + str(dig) + str(imgno+1) + ".jpeg"
        # print(name)
        im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
        arr = np.asarray(colr)
        arr = arr.astype(int)
        images_train.append(arr)
        cls_train.append(dig)
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l[dig] = 1
        labels_train.append(l)

# operator = ['+', 'x', 'd', 'e']
# for i in range(0, 90):
#     for oper in range(0, 4):
#         name = "ShivamTestdata/" + str(operator[oper]) + "/fo" + str(operator[oper]) + str(i+1) + ".jpeg"
#         # print(name)
#         im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
#         colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
#         arr = np.asarray(colr)
#         arr = arr.astype(int)
#         images_train.append(arr)
#         cls_train.append(oper+10)
#         l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         l[oper+10] = 1
#         labels_train.append(l)

images_train = np.asarray(images_train, dtype=np.float32)
cls_train = np.array(cls_train)
labels_train = np.asarray(labels_train, dtype=np.float32)

model = inception.Inception()

file_path_cache_train = os.path.join('inception_owndata_train.pkl')
images_scaled = images_train #* 255.0

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

transfer_len = model.transfer_len
num_classes = 14

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Wrap the transfer-values as a Pretty Tensor object.
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 20
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)
            saver.save(sess=session, save_path=save_path)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


print(model)
print(session)
hyperparams = {'x' : x, 'y_true_cls' : y_true_cls, 'y_true' : y_true, 'y_pred_cls' : y_pred_cls}
optimize(num_iterations=1000)


pts = deque()
counter = 40
while (True):
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
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
    		plotting(list(pts), model, session, saver, save_path, hyperparams)
    		pts.clear()
    	cv2.imshow('frame',frame)
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break

camera.release()
cv2.destroyAllWindows()