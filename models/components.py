import tensorflow as tf

def downsample(
        inputs,
        filters,
        rate,
        kernel_size,
        conv_stride,
        bn,
        training,
        trainable,
        n_conv=2,
        suffix='',
):
    """down sampling block"""
    with tf.variable_scope('downsample{}'.format(suffix)):
        conv = inputs
        for i in range(n_conv):
            conv = tf.layers.conv2d(
                inputs=conv, filters=filters, kernel_size=kernel_size,
                strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
            if bn:
                conv = tf.layers.batch_normalization(conv, training=training)
        half = tf.layers.max_pooling2d(conv, rate, rate)
    return conv, half

def upsample(
        inputs,
        reference,
        filters,
        rate,
        kernel_size,
        conv_stride,
        bn,
        training,
        trainable,
        suffix='',
):
    """up sampling block"""
    with tf.variable_scope('upsample{}'.format(suffix)):
        reference_size = int(reference.get_shape()[1])

        tconv0 = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=rate, strides=rate,
            padding='valid', activation=None, trainable=trainable)
        if bn:
            tconv0 = tf.layers.batch_normalization(tconv0, training=training)
        tconv0_size = int(tconv0.get_shape()[1])

        print('inputs:{}'.format(inputs.get_shape()))
        print('tconv:{}'.format(tconv0.get_shape()))
        print('reference:{}'.format(reference.get_shape()))

        # assuming reference_size > tconv0_size
        assert reference_size >= tconv0_size, '{} >= {}'.format(reference_size, tconv0_size)
        diff = reference_size - tconv0_size
        diff_half = tf.cast(diff/2, tf.int32)

        concatenated = tf.concat([tconv0, tf.image.crop_to_bounding_box(
            reference, diff_half, diff_half, tconv0_size, tconv0_size)], axis=-1)
        print('concatenated:{}'.format(concatenated.get_shape()))

        conv0 = tf.layers.conv2d(
            inputs=concatenated, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
        if bn:
            conv0 = tf.layers.batch_normalization(conv0, training=training)
        conv1 = tf.layers.conv2d(
            inputs=conv0, filters=filters, kernel_size=kernel_size,
            strides=conv_stride, padding='valid', activation=tf.nn.relu, trainable=trainable)
        if bn:
            conv1 = tf.layers.batch_normalization(conv1, training=training)
    return conv1

def encoder(inputs, filters_first, n_downsample, rate, kernel_size, conv_stride, bn, training, trainable, n_conv=2):
    """encoder block"""
    assert (not bn) or (training is not None), 'Error: Training is required in BN'
    res_list = list()
    next_inputs = inputs
    next_filters = filters_first

    with tf.variable_scope('encoder'):
        for i in range(n_downsample):
            print(next_inputs.get_shape())
            res, downsampled = downsample(
                inputs=next_inputs,
                filters=next_filters,
                rate=rate,
                kernel_size=kernel_size,
                conv_stride=conv_stride,
                n_conv=n_conv,
                bn=bn,
                training=training,
                trainable=trainable,
                suffix=i,
            )
            res_list.append(res)

            next_inputs = downsampled
            next_filters = int(rate * next_filters)

    return res_list, downsampled

def decoder(inputs, res_list, rate, kernel_size, conv_stride, bn, training, trainable):
    """decoder block"""
    assert (not bn) or (training is not None), 'Error: Training is required in BN'
    filters_first = inputs.get_shape()[-1]
    next_inputs = inputs
    next_filters = filters_first

    with tf.variable_scope('decoder'):
        for i in range(len(res_list)):
            print(next_inputs.get_shape())
            upsampled = upsample(
                inputs=next_inputs,
                reference=res_list[-1],
                filters=next_filters,
                rate=rate,
                kernel_size=kernel_size,
                conv_stride=conv_stride,
                bn=bn,
                training=training,
                trainable=trainable,
                suffix=i,
            )

            next_inputs = upsampled
            next_filters = int(int(next_filters)/2)
            del res_list[-1]

    return upsampled

def unet(
        inputs,
        filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        training=None,
        trainable=True,
):
    '''
    this func represents unet
    '''
    assert (not bn) or (training is not None), 'Error: Training is required in BN'
    with tf.variable_scope('unet'):
        res_list, downsampled = encoder(
            inputs=inputs,
            filters_first=filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            training=training,
            trainable=trainable,
        )
        output = decoder(
            inputs=downsampled,
            res_list=res_list,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            training=training,
            trainable=trainable,
        )

    # res_list must be empty because all of them are supposed to be consumed
    assert not res_list
    return output

def unet_based_annotator(
        input_,
        n_filters_first,
        n_downsample,
        rate,
        kernel_size,
        conv_stride,
        bn=False,
        training=None,
        trainable=True
):
    '''
    this function represents a part of model which will produce annotation or segmentation
    Args:
        input_: input tensor
        n_filters_first: the num of filters in the first block
        n_downsample: the num of downsample
        rate: the rate of downsample and upsample
        kernel_size: kernel_size of every Conv
        conv_stride: stride in conv
        bn: (bool) whether or not BN is applied
        training: (bool) whether the model is being trained
            this can be None as long as bn=False
        trainable: (bool) whether or not this block is trainable
    '''
    assert (not bn) or (training is not None), 'Error: Training is required in BN'

    with tf.variable_scope('annotator'):
        unet_out = unet(
            inputs=input_,
            filters_first=n_filters_first,
            n_downsample=n_downsample,
            rate=rate,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            bn=bn,
            training=training,
            trainable=trainable,
        )
        seg = tf.layers.conv2d(inputs=unet_out, filters=1, kernel_size=1, activation=None, trainable=trainable)
    return seg
