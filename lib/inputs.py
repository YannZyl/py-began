# -*- coding: utf-8 -*-
import cv2
import os.path as osp
import tensorflow as tf
from glob import glob

def data_loader(data_dir, batch_size, image_scale):
    # get file name queue 
    filename_list = glob(osp.join(data_dir,'*.jpg'))
    filename_queue = tf.train.string_input_producer(filename_list)
    # get image shape
    im = cv2.imread(filename_list[0])
    h, w, _ = im.shape
    # read file
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    # decode image and reshape
    image = tf.image.decode_jpeg(data, channels=3)
    image.set_shape([h,w,3])
    
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    batch_image = tf.train.shuffle_batch([image], batch_size, capacity, min_after_dequeue, num_threads=4)
    
    # crop and resize
    batch_image = tf.image.crop_to_bounding_box(batch_image,50,25,128,128)
    batch_image = tf.image.resize_nearest_neighbor(batch_image, [image_scale,image_scale])
    
    # nromalized to [-1,1]
    batch_image = tf.cast(batch_image, tf.float32)
    batch_image = batch_image / 127.5 - 1.0
    return batch_image
