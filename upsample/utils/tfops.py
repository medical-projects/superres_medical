'''
provide various additional tf ops
'''

import tensorflow as tf

def decode_image(encoded_image, determine_shape_=True, skip_read=False):
    '''
    decode image

    Args:
        encoded_image: string encoded image or image path
        determine_shape: whether or not this func should determine the shape
        skip_read: if encoded_image is a string encoded image,
            then set this to True.
            otherwise, set this to False so that this func will
            apply tf.read first.
    '''
    if not skip_read: encoded_image = tf.read_file(encoded_image)
    image = tf.image.decode_image(encoded_image)
    if determine_shape_:
        # image = determine_image_shape(image)
        image = determine_image_channel(image, 1)
    return image

def determine_image_channel(tensor, channel=1):
    '''
    detemrine the image color channel
    '''
    shape = tensor.get_shape()
    tensor.set_shape([*shape[:2], channel])
    return tensor

def determine_image_shape(tensor):
    '''
    determine tensor shape
    '''
    shape = tf.shape(tensor)
    tensor = tf.image.resize_image_with_crop_or_pad(tensor, shape[0], shape[1])
    return tensor

def image_central_crop_boundingbox(tensor, target_shape):
    '''
    crop central part of image according to target_shape

    tensor is supposed to be 4D [batch, height, width, channel]
    '''
    current_shape = tf.shape(tensor)

    offset_width = (current_shape[1] - target_shape[0]) // 2
    offset_height = (current_shape[2] - target_shape[1]) // 2
    cropped = tf.image.crop_to_bounding_box(
        tensor,
        offset_width=offset_width,
        offset_height=offset_height,
        target_width=target_shape[0],
        target_height=target_shape[1],
    )
    print(tensor)
    print(cropped)
    exit(0)

    with tf.control_dependencies([
            tf.assert_greater_equal(
                current_shape[1:2],
                target_shape[0:1],
                data=[current_shape, target_shape]
            )
    ]):
        output = tf.cond(
            tf.reduce_all(tf.equal(current_shape[1:2], target_shape[0:1])),
            true_fn=lambda: tensor,
            false_fn=lambda: cropped,
        )
    return output

def dict_map(dict_in, key_source, key_target, op):
    '''
    apply op to elements in a dataset
    each element is supposed to be a dict

    element[key_target] = op(element[key_source])
    '''
    dict_in[key_target] = op(dict_in[key_source])

    return dict_in

def scale_image(image, method='bicubic', scale=0.5):
    '''
    downsample an image
    '''
    method = method.lower()

    if method in ('bilinear', 'nearest_neighbor', 'bicubic', 'area'):
        new_size = tf.cast(tf.multiply(tf.cast(tf.shape(image)[:2], tf.float32), tf.cast(scale, tf.float32)), tf.int32)
        scaled_image = tf.image.resize_images(image, size=new_size)
    elif method == 'fft':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return scaled_image

def dict_split(dict_, split_target):
    '''
    for a given dict, this func will take values
    corresponding to specified keys (split_target),
    and make a sencond dict which holds extracted values.

    return value will be a tuple(dict_, new_dict)
    '''
    new_dict = {key: dict_[key] for key in split_target}
    for key in split_target: del(dict_[key])
    return (dict_, new_dict)
