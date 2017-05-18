from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import time
from datetime import timedelta
import os
from tensorflow.examples.tutorials.mnist import input_data

def predict():
  # data = input_data.read_data_sets('MNIST_data', one_hot=True)
  # data.train.cls = np.argmax(data.train.labels, axis=1)
  d = []
  t = []
  label_val = []
  # for i in range(len(data.train.images)):
  #     if(i%1000 == 0):
  #         print(i)
  #     images = data.train.images[i]
  #     temp=np.reshape(images,(28,28))
  #     arr = np.asarray(temp)
  #     arr = np.ceil(arr)
  #     arr = arr.astype(int)
  #     arr = arr*255
  #     cv2.imwrite('foo1.jpeg', arr)
  #     images = cv2.imread("foo1.jpeg", cv2.IMREAD_GRAYSCALE)
  #     (thresh, im_bw) = cv2.threshold(images, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  #     im_bw = cv2.threshold(images, thresh, 255, cv2.THRESH_BINARY)[1]
  #     arr = im_bw
  #     _, contours, hier = cv2.findContours(im_bw.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  #     cnt = contours[-1]
  #     x,y,w,h = cv2.boundingRect(cnt)
  #     arr = arr[y:y+h, x:x+w]
  #     arr = cv2.resize(arr, (28, 28))
  #     arr = np.reshape(arr, 784)
  #     arr = arr/255
  #     d.append(arr)
  #     t.append(data.train.cls[i])
  #     label_val.append(data.train.labels[i].astype(int))

  d = np.asarray(d, dtype=np.float32)
  t = np.array(t)
  label_val = np.asarray(label_val, dtype=np.float32)
  # Convolutional Layer 1.
  filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
  num_filters1 = 16         # There are 16 of these filters.

  # Convolutional Layer 2.
  filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
  num_filters2 = 36         # There are 36 of these filters.

  # Fully-connected layer.
  fc_size = 128             # Number of neurons in fully-connected layer.

  # We know that MNIST images are 28 pixels in each dimension.
  img_size = 28

  # Images are stored in one-dimensional arrays of this length.
  img_size_flat = img_size * img_size

  # Tuple with height and width of images used to reshape arrays.
  img_shape = (img_size, img_size)

  # Number of colour channels for the images: 1 channel for gray-scale.
  num_channels = 1

  # Number of classes, one class for each of 10 digits.
  num_classes = 10

  eval_data = []
  test_label = []
  test_label_hot = []
  for i in range(0, 1):
    name = "foo" + str(i+1) + ".jpeg"
    im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.bitwise_not(im_gray)
    im_gray = im_gray/255
    arr = np.asarray(im_gray)
    arr = arr.astype(int)
    for i in range(1, 27, 2):
      for j in range(1, 27, 2):
          l1 = []
          l0 = []
          if(arr[i][j] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i][j+1] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i+1][j] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i+1][j+1] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(len(l1) >= len(l0)):
              arr[i][j] = 1
              arr[i+1][j] = 1
              arr[i][j+1] = 1
              arr[i+1][j+1] = 1
          else:
              arr[i][j] = 0
              arr[i+1][j] = 0
              arr[i][j+1] = 0
              arr[i+1][j+1] = 0

    for i in range(2, 27, 2):
      for j in range(2, 27, 2):
          l1 = []
          l0 = []
          if(arr[i][j] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i][j+1] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i+1][j] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(arr[i+1][j+1] == 1):
              l1.append(1)
          else:
              l0.append(1)
          if(len(l1) >= len(l0)):
              arr[i][j] = 1
              arr[i+1][j] = 1
              arr[i][j+1] = 1
              arr[i+1][j+1] = 1
          else:
              arr[i][j] = 0
              arr[i+1][j] = 0
              arr[i][j+1] = 0
              arr[i+1][j+1] = 0
    # arr = arr/255.0
    arr = np.reshape(arr, 784)
    eval_data.append(arr)
    test_label.append(0)
    test_label_hot.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  eval_data = np.asarray(eval_data, dtype=np.float32)
  test_label = np.array(test_label)
  test_label_hot = np.asarray(test_label_hot, dtype=np.float32)

  def get_weights_variable(layer_name):
      # Retrieve an existing variable named 'weights' in the scope
      # with the given layer_name.
      # This is awkward because the TensorFlow function was
      # really intended for another purpose.

      with tf.variable_scope(layer_name, reuse=True):
          variable = tf.get_variable('weights')

      return variable

  def new_weights(shape):
      return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

  def new_biases(length):
      return tf.Variable(tf.constant(0.05, shape=[length]))

  def new_conv_layer(input,              # The previous layer.
                     num_input_channels, # Num. channels in prev. layer.
                     filter_size,        # Width and height of each filter.
                     num_filters,        # Number of filters.
                     use_pooling=True):  # Use 2x2 max-pooling.

      # Shape of the filter-weights for the convolution.
      # This format is determined by the TensorFlow API.
      shape = [filter_size, filter_size, num_input_channels, num_filters]

      # Create new weights aka. filters with the given shape.
      weights = new_weights(shape=shape)

      # Create new biases, one for each filter.
      biases = new_biases(length=num_filters)

      # Create the TensorFlow operation for convolution.
      # Note the strides are set to 1 in all dimensions.
      # The first and last stride must always be 1,
      # because the first is for the image-number and
      # the last is for the input-channel.
      # But e.g. strides=[1, 2, 2, 1] would mean that the filter
      # is moved 2 pixels across the x- and y-axis of the image.
      # The padding is set to 'SAME' which means the input image
      # is padded with zeroes so the size of the output is the same.
      layer = tf.nn.conv2d(input=input,
                           filter=weights,
                           strides=[1, 1, 1, 1],
                           padding='SAME')

      # Add the biases to the results of the convolution.
      # A bias-value is added to each filter-channel.
      layer += biases

      # Use pooling to down-sample the image resolution?
      if use_pooling:
          # This is 2x2 max-pooling, which means that we
          # consider 2x2 windows and select the largest value
          # in each window. Then we move 2 pixels to the next window.
          layer = tf.nn.max_pool(value=layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')

      # Rectified Linear Unit (ReLU).
      # It calculates max(x, 0) for each input pixel x.
      # This adds some non-linearity to the formula and allows us
      # to learn more complicated functions.
      layer = tf.nn.relu(layer)

      # Note that ReLU is normally executed before the pooling,
      # but since relu(max_pool(x)) == max_pool(relu(x)) we can
      # save 75% of the relu-operations by max-pooling first.

      # We return both the resulting layer and the filter-weights
      # because we will plot the weights later.
      return layer, weights

  def flatten_layer(layer):
      # Get the shape of the input layer.
      layer_shape = layer.get_shape()

      # The shape of the input layer is assumed to be:
      # layer_shape == [num_images, img_height, img_width, num_channels]

      # The number of features is: img_height * img_width * num_channels
      # We can use a function from TensorFlow to calculate this.
      num_features = layer_shape[1:4].num_elements()
      
      # Reshape the layer to [num_images, num_features].
      # Note that we just set the size of the second dimension
      # to num_features and the size of the first dimension to -1
      # which means the size in that dimension is calculated
      # so the total size of the tensor is unchanged from the reshaping.
      layer_flat = tf.reshape(layer, [-1, num_features])

      # The shape of the flattened layer is now:
      # [num_images, img_height * img_width * num_channels]

      # Return both the flattened layer and the number of features.
      return layer_flat, num_features


  def new_fc_layer(input,          # The previous layer.
                   num_inputs,     # Num. inputs from prev. layer.
                   num_outputs,    # Num. outputs.
                   use_relu=True): # Use Rectified Linear Unit (ReLU)?

      # Create new weights and biases.
      weights = new_weights(shape=[num_inputs, num_outputs])
      biases = new_biases(length=num_outputs)

      # Calculate the layer as the matrix multiplication of
      # the input and weights, and then add the bias-values.
      layer = tf.matmul(input, weights) + biases

      # Use ReLU?
      if use_relu:
          layer = tf.nn.relu(layer)

      return layer

  # CNN

  x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
  x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
  y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
  y_true_cls = tf.argmax(y_true, dimension=1)

  layer_conv1, weights_conv1 = \
      new_conv_layer(input=x_image,
                     num_input_channels=num_channels,
                     filter_size=filter_size1,
                     num_filters=num_filters1,
                     use_pooling=True)

  layer_conv2, weights_conv2 = \
      new_conv_layer(input=layer_conv1,
                     num_input_channels=num_filters1,
                     filter_size=filter_size2,
                     num_filters=num_filters2,
                     use_pooling=True)

  layer_flat, num_features = flatten_layer(layer_conv2)

  layer_fc1 = new_fc_layer(input=layer_flat,
                           num_inputs=num_features,
                           num_outputs=fc_size,
                           use_relu=True)

  layer_fc2 = new_fc_layer(input=layer_fc1,
                           num_inputs=fc_size,
                           num_outputs=num_classes,
                           use_relu=False)

  y_pred = tf.nn.softmax(layer_fc2)
  y_pred_prob = tf.reduce_max(y_pred, axis=1)*100
  y_pred_cls = tf.argmax(y_pred, dimension=1)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                          labels=y_true) 	

  cost = tf.reduce_mean(cross_entropy)

  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  save_dir = 'checkpoints/'
  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
  save_path = os.path.join(save_dir, 'best_validation')

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  train_batch_size = 64

  # Counter for total number of iterations performed so far.
  total_iterations = 0
  k = 0
  def optimize(num_iterations):
      # Ensure we update the global variable rather than a local copy.
      global total_iterations
      global k
      # Start-time used for printing time-usage below.
      start_time = time.time()

      for i in range(total_iterations,
                     total_iterations + num_iterations):

          # Get a batch of training examples.
          # x_batch now holds a batch of images and
          # y_true_batch are the true labels for those images.
          # x_batch, y_true_batch = data.train.next_batch(train_batch_size)

          if(i+k > 55000):
              k = 0

          x_batch = d[i+k:i+k+train_batch_size]
          y_true_batch = label_val[i+k:i+k+train_batch_size]

          k = k + train_batch_size
          # Put the batch into a dict with the proper names
          # for placeholder variables in the TensorFlow graph.
          feed_dict_train = {x: x_batch,
                             y_true: y_true_batch}

          # Run the optimizer using this batch of training data.
          # TensorFlow assigns the variables in feed_dict_train
          # to the placeholder variables and then runs the optimizer.
          session.run(optimizer, feed_dict=feed_dict_train)
          saver.save(sess=session, save_path=save_path)

          # Print status every 100 iterations.
          if i % 100 == 0:
              # Calculate the accuracy on the training-set.
              acc = session.run(accuracy, feed_dict=feed_dict_train)

              # Message for printing.
              msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

              # Print it.
              print(msg.format(i + 1, acc))

      # Update the total number of iterations performed.
      total_iterations += num_iterations

      # Ending time.
      end_time = time.time()

      # Difference between start and end-times.
      time_dif = end_time - start_time

      # Print the time-usage.
      print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

  # start_time = time.time()
  # for i in range(50):
  #     feed_dict_train = {x : d, y_true : label_val, y_true_cls : t}
  #     session.run(optimizer, feed_dict=feed_dict_train)
  #     saver.save(sess=session, save_path=save_path)
  # end_time = time.time()

  # optimize(num_iterations=10000)
  # eval_data = []
  # test_label = []
  # test_label_hot = []
  # name = "final1.jpeg"
  # im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
  # arr = np.asarray(image)
  # arr = np.reshape(arr, 784)
  # eval_data.append(arr)
  # eval_data = np.asarray(eval_data, dtype=np.float32)
  # test_label.append(0)
  # eval_labels = np.array(test_label)
  # test_label_hot.append(([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
  # eval_label_hot = np.asarray(test_label_hot, dtype=np.float32)

  saver.restore(sess=session, save_path=save_path)
  feed_dict_test = {x:eval_data, y_true_cls:test_label, y_true : test_label_hot}
  cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
  cls_prob = session.run(y_pred_prob, feed_dict=feed_dict_test)
  l0 = []
  for i in range(0, 1):
      l0.append((cls_pred[i], cls_prob[i]))

  print(l0)
  tf.reset_default_graph()
  session.close()