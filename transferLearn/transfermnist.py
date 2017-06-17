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

data = input_data.read_data_sets('MNIST_data', one_hot=True)
data.train.cls = np.argmax(data.train.labels, axis=1)
data.test.cls = np.argmax(data.test.labels, axis=1)
d = []
t = []
label_val = []
for i in range(len(data.train.images)):
    if(i%1000 == 0):
        print(i)
    images = data.train.images[i]
    temp=np.reshape(images,(28,28))
    arr = np.asarray(temp)
    arr = np.ceil(arr)
    arr = arr.astype(int)
    arr = arr*255
    cv2.imwrite('foo1.jpeg', arr)
    images = cv2.imread("foo1.jpeg", cv2.IMREAD_GRAYSCALE)
    colr = cv2.cvtColor(images,cv2.COLOR_GRAY2RGB)
    arr = np.asarray(colr)
    arr = arr.astype(int)
    # (thresh, im_bw) = cv2.threshold(images, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # im_bw = cv2.threshold(images, thresh, 255, cv2.THRESH_BINARY)[1]
    # arr = im_bw
    # _, contours, hier = cv2.findContours(im_bw.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[-1]
    # x,y,w,h = cv2.boundingRect(cnt)
    # arr = arr[y:y+h, x:x+w]
    # arr = cv2.resize(arr, (28, 28))
    # arr = np.reshape(arr, 784)
    # arr = arr/255
    d.append(arr)
    t.append(data.train.cls[i])
    label_val.append(data.train.labels[i].astype(int))

images_train = np.asarray(d, dtype=np.float32)
cls_train = np.array(t)
labels_train = np.asarray(label_val, dtype=np.float32)

d = []
t = []
label_val = []
for i in range(len(data.test.labels)):
    if(i%1000 == 0):
        print(i)
    images = data.test.images[i]
    temp=np.reshape(images,(28,28))
    arr = np.asarray(temp)
    arr = np.ceil(arr)
    arr = arr.astype(int)
    arr = arr*255
    cv2.imwrite('foo1.jpeg', arr)
    images = cv2.imread("foo1.jpeg", cv2.IMREAD_GRAYSCALE)
    colr = cv2.cvtColor(images,cv2.COLOR_GRAY2RGB)
    arr = np.asarray(colr)
    arr = arr.astype(int)
    # cv2.imwrite('foo1.jpeg', arr)
    # images = cv2.imread("foo1.jpeg", cv2.IMREAD_GRAYSCALE)
    # (thresh, im_bw) = cv2.threshold(images, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # im_bw = cv2.threshold(images, thresh, 255, cv2.THRESH_BINARY)[1]
    # arr = im_bw
    # _, contours, hier = cv2.findContours(im_bw.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[-1]
    # x,y,w,h = cv2.boundingRect(cnt)
    # arr = arr[y:y+h, x:x+w]
    # arr = cv2.resize(arr, (28, 28))
    # arr = np.reshape(arr, 784)
    # arr = arr/255
    d.append(arr)
    t.append(data.test.cls[i])
    label_val.append(data.test.labels[i].astype(int))

images_test = np.asarray(d, dtype=np.float32)
cls_test = np.array(t)
labels_test = np.asarray(label_val, dtype=np.float32)

# images_train = []
# cls_train = []
# labels_train = []

# for num in range(0, 10):
# 	for i in range(0, 40):
# 		name = "FinalTest/" + str(num) + "/final" + str(i+1) + ".jpeg"
# 		# print(name)
# 		im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# 		# im_gray = cv2.bitwise_not(im_gray)
# 		# im_gray = im_gray/255
# 		# arr = np.asarray(im_gray)
# 		# arr = arr.astype(int)
# 		# for i in range(1, 27, 2):
# 		# 	for j in range(1, 27, 2):
# 		# 		l1 = []
# 		# 		l0 = []
# 		# 		if(arr[i][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(len(l1) >= len(l0)):
# 		# 			arr[i][j] = 1
# 		# 			arr[i+1][j] = 1
# 		# 			arr[i][j+1] = 1
# 		# 			arr[i+1][j+1] = 1
# 		# 		else:
# 		# 			arr[i][j] = 0
# 		# 			arr[i+1][j] = 0
# 		# 			arr[i][j+1] = 0
# 		# 			arr[i+1][j+1] = 0

# 		# for i in range(2, 27, 2):
# 		# 	for j in range(2, 27, 2):
# 		# 		l1 = []
# 		# 		l0 = []
# 		# 		if(arr[i][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(len(l1) >= len(l0)):
# 		# 			arr[i][j] = 1
# 		# 			arr[i+1][j] = 1
# 		# 			arr[i][j+1] = 1
# 		# 			arr[i+1][j+1] = 1
# 		# 		else:
# 		# 			arr[i][j] = 0
# 		# 			arr[i+1][j] = 0
# 		# 			arr[i][j+1] = 0
# 		# 			arr[i+1][j+1] = 0
# 		# arr = arr/255.0
# 		# arr = np.reshape(arr, 784)
# 		colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
# 		arr = np.asarray(colr)
# 		arr = arr.astype(int)
# 		images_train.append(arr)
# 		cls_train.append(num)
# 		l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 		l[num] = 1
# 		labels_train.append(l)

# images_train = np.asarray(images_train, dtype=np.float32)
# cls_train = np.array(cls_train)
# labels_train = np.asarray(labels_train, dtype=np.float32)


# images_test = []
# cls_test = []
# labels_test = []
# for num in range(0, 10):
# 	for i in range(40, 50):
# 		name = "FinalTest/" + str(num) + "/final" + str(i+1) + ".jpeg"
# 		im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
# 		# im_gray = cv2.bitwise_not(im_gray)
# 		# im_gray = im_gray/255
# 		# arr = np.asarray(im_gray)
# 		# arr = arr.astype(int)
# 		# for i in range(1, 27, 2):
# 		# 	for j in range(1, 27, 2):
# 		# 		l1 = []
# 		# 		l0 = []
# 		# 		if(arr[i][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(len(l1) >= len(l0)):
# 		# 			arr[i][j] = 1
# 		# 			arr[i+1][j] = 1
# 		# 			arr[i][j+1] = 1
# 		# 			arr[i+1][j+1] = 1
# 		# 		else:
# 		# 			arr[i][j] = 0
# 		# 			arr[i+1][j] = 0
# 		# 			arr[i][j+1] = 0
# 		# 			arr[i+1][j+1] = 0

# 		# for i in range(2, 27, 2):
# 		# 	for j in range(2, 27, 2):
# 		# 		l1 = []
# 		# 		l0 = []
# 		# 		if(arr[i][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(arr[i+1][j+1] == 1):
# 		# 			l1.append(1)
# 		# 		else:
# 		# 			l0.append(1)
# 		# 		if(len(l1) >= len(l0)):
# 		# 			arr[i][j] = 1
# 		# 			arr[i+1][j] = 1
# 		# 			arr[i][j+1] = 1
# 		# 			arr[i+1][j+1] = 1
# 		# 		else:
# 		# 			arr[i][j] = 0
# 		# 			arr[i+1][j] = 0
# 		# 			arr[i][j+1] = 0
# 		# 			arr[i+1][j+1] = 0
# 		# arr = arr/255.0
# 		# arr = np.reshape(arr, 784)
# 		colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
# 		arr = np.asarray(colr)
# 		arr = arr.astype(int)
# 		images_test.append(arr)
# 		cls_test.append(num)
# 		l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 		l[num] = 1
# 		labels_test.append(l)

# images_test = np.asarray(images_test, dtype=np.float32)
# cls_test = np.array(cls_test)
# labels_test = np.asarray(labels_test, dtype=np.float32)

inception.maybe_download()
model = inception.Inception()

file_path_cache_train = os.path.join('inception_mnist_train.pkl')
file_path_cache_test = os.path.join('inception_mnist_test.pkl')

print("Processing Inception transfer-values for training-images ...")

images_scaled = images_train #* 255.0

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")


images_scaled = images_test #* 255.0

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

print(transfer_values_train.shape)
print(transfer_values_test.shape)                                             


transfer_len = model.transfer_len
num_classes = 10

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

train_batch_size = 64



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

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))



optimize(num_iterations=15000)

feed_dict_test = {x:transfer_values_test[0:10000], y_true_cls:cls_test[0:10000], y_true : labels_test[0:10000]}
cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
pred = session.run(accuracy, feed_dict=feed_dict_test)

# print(cls_pred)
print(pred)