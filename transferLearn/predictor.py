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

def predict(model, session, saver, save_path, hyperparams):
    images_test = []
    cls_test = []
    labels_test = []
    name = "foo1.jpeg"
    im_gray = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    colr = cv2.cvtColor(im_gray,cv2.COLOR_GRAY2RGB)
    arr = np.asarray(colr)
    arr = arr.astype(int)
    images_test.append(arr)
    cls_test.append(0)
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels_test.append(l)

    images_test = np.asarray(images_test, dtype=np.float32)
    cls_test = np.array(cls_test)
    labels_test = np.asarray(labels_test, dtype=np.float32)

    file_path_cache_train = os.path.join('inception_owndata_train.pkl')
    file_path_cache_test = os.path.join('inception_owndata_test.pkl')

    images_scaled = images_test #* 255.0

    transfer_values_test = tran
    sfer_values_cache(cache_path=file_path_cache_test,
                                                 images=images_scaled,
                                                 model=model)
    saver.restore(sess=session, save_path=save_path)
    feed_dict_test = {hyperparams['x']:transfer_values_test, hyperparams['y_true_cls']:cls_test, hyperparams['y_true'] : labels_test}
    cls_pred = session.run(hyperparams['y_pred_cls'], feed_dict=feed_dict_test)
    # pred = session.run(accuracy, feed_dict=feed_dict_test)

    print(cls_pred) 






