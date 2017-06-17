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
import random
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

images_train = []
cls_train = []
labels_train = []

k = -1
for char in alphabet:
    k = k + 1
    for j in range(0, min(len(os.listdir('../BtpAlpha/images/' + str(char)))-1, 1000)):
        if(j%1000 == 0):
            print(j)
        name = "../BtpAlpha/images/" + str(char) + "/foo" + str(j+1) + ".jpeg"
    	# print(name)
        im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
        arr = np.asarray(colr)
        arr = arr.astype(int)
        images_train.append(arr)
        cls_train.append(k)
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l[k] = 1
        labels_train.append(l)


# RANDOM SHUFFLING OF TRAINING DATA
d_shuffle = []
t_shuffle = []
label_val_shuffle = []
index_shuf = list(range(len(images_train)))
random.shuffle(index_shuf)
for i in index_shuf:
    d_shuffle.append(images_train[i])
    t_shuffle.append(cls_train[i])
    label_val_shuffle.append(labels_train[i])
images_train = d_shuffle
cls_train = t_shuffle
labels_train = label_val_shuffle

images_train = np.asarray(images_train, dtype=np.float32)
cls_train = np.array(cls_train)
labels_train = np.asarray(labels_train, dtype=np.float32)


images_test = []
cls_test = []
labels_test = []

k = -1
for char in alphabet:
    k = k + 1
    for j in range(1000, min(len(os.listdir('../BtpAlpha/images/' + str(char)))-1, 1600)):
        if(j%1000 == 0):
            print(j)
        name = "../BtpAlpha/images/" + str(char) + "/foo" + str(j+1) + ".jpeg"
        im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
        arr = np.asarray(colr)
        arr = arr.astype(int)
        images_test.append(arr)
        cls_test.append(k)
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l[k] = 1
        labels_test.append(l)

images_test = np.asarray(images_test, dtype=np.float32)
cls_test = np.array(cls_test)
labels_test = np.asarray(labels_test, dtype=np.float32)

inception.maybe_download()
model = inception.Inception()

file_path_cache_train = os.path.join('inception_alpha_train.pkl')
file_path_cache_test = os.path.join('inception_alpha_test.pkl')

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
num_classes = 26

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



optimize(num_iterations=10000)

feed_dict_test = {x:transfer_values_test, y_true_cls:cls_test, y_true : labels_test}
cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
pred = session.run(accuracy, feed_dict=feed_dict_test)

for i in range(0, 100):
    print(cls_pred[i])
# print(cls_pred)
print(pred)





