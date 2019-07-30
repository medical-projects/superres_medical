
"""
module to provide additional tf ops
"""
import random
from operator import add
import regex
import numpy as np
import urllib
import tensorflow as tf
from functools import reduce
from tensorflow.python.client import device_lib
import os
import multiprocessing as mp
import cv2
import wget
import pandas as pd
import matplotlib.pyplot as plt

def tf_threshold(input_tensor, threshold, max_val, dtype=tf.int64):
    '''
    this function will performs thresholding operation
    for input tensor. elements with larger or equal to
    threshold will be max_val otherwise 0
    '''
    output = tf.cast(tf.greater_equal(input_tensor, threshold), dtype)*max_val
    output = tf.cast(output, dtype)
    return output

def tf_exists(target_file_list):
    '''
    this function will check if all the
    files in target_file_list exist
    this function will return True only
    if ALL the files exist
    '''
    def py_exists(targets):
        '''
        this function actually
        performs the operation
        '''
        for target in targets:
            if not tf.gfile.Exists(target):
                print('WARNING: {} does not exists'.format(target))
                print('WARNING: list : {}\n'.format(targets))
                return False
        return True

    return tf.py_func(
        py_exists, [target_file_list], tf.bool
    )

def tf_determine_image_channel(image, channel_size):
    '''
    this function will determine the
    chennel of a image, which is necessary
    when you are using conv, since they
    require channel size to initialize
    their filter
    Args:
        image: input image
        channel_size: channel size
            3 for 3 channel image
            1 for mono-chromatic image
            other for custom images
    Return:
        image with channel size determined
    '''
    new_shape = list(image.get_shape())

    new_shape[-1] = channel_size
    image.set_shape(new_shape)
    return image

def tf_determine_image_size(image):
    '''
    normally, tf doesn't know the size of
    images and it causes various problems
    sometimes. this developers can use 
    this function to determine them.
    '''
    return tf.image.resize_image_with_crop_or_pad(image, tf.shape(image)[0], tf.shape(image)[1])

def tf_extract_label(image, label_value, gray_scale=True):
    """
    this function extracts label image
    from given image which contains
    background and label on it.
    Args:
        image: input image, must be 1 channel
        label_value: pixel value of label
    Returns:
        image that contains only image
    """
    label_image = tf.cast(tf.equal(image, label_value), image.dtype) * 255

    if gray_scale:
        label_image = label_image[:, :, 0:1]

    return label_image

