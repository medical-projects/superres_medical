'''
provide various additional tf ops
'''

import tensorflow as tf

def image_central_crop_boundingbox(tensor, target_shape):
    '''
    crop central part of image according to target_shape

    tensor is supposed to be 4D [batch, height, width, channel]
    '''
    current_shape = tf.shape(tensor)

    offset_height = (current_shape[1] - target_shape[1]) // 2
    offset_width = (current_shape[2] - target_shape[2]) // 2
    cropped = tf.image.crop_to_bounding_box(
        tensor,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_shape[1],
        target_width=target_shape[1],
    )

    with tf.control_dependencies([
            tf.assert_greater_eual(
                current_shape[1:2],
                target_shape[1:2],
                data=[current_shape, target_shape]
            )
    ]):
        output = tf.cond(
            tf.equal(current_shape[1:2], target_shape[1:2]),
            true_fn=lambda: tensor,
            false_fn=lambda: cropped,
        )
    return output
