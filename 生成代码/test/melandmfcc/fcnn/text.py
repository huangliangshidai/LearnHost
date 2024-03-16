# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
