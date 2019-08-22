'''
GAN-CIRCLE without gan structure
'''
import tensorflow as tf
from upsample.models import components
from functools import partial
from upsample.utils import tfops
import os

default_params = {
    'model': 'gan_circle_nongan',
    'add_bicubic': True,
    'feature_extract_init_filteres': 64,
    'feature_extract_final_filteres': 16,
    'feature_extract_filter_step': 3,
    'upsample_scale': 2,
    'batch_size': 2,
}

def unit_block(input_, filters, kernel_size=3):
    output = input_
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(input_.get_shape)
    output = tf.layers.conv2d(
        output,
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.leaky_relu,
    )

    with tf.control_dependencies([tf.print(tf.shape(output))]):
        return output
    print(output.get_shape())
    exit(0)
    return output

def model(features, labels, mode, params, config):
    '''model def'''
    scale = params['upsample_scale']
    lrimage = features['lrimage']

    ndim = len(lrimage.get_shape())

    if ndim == 4:
        original_shape = tf.shape(lrimage)
        target_size = original_shape[1:-1] * scale
    elif ndim == 3:
        target_size = tf.shape(lrimage)[1:] * scale
    else:
        RuntimeError()

    features = components.semi_densenet(
        input_=lrimage,
        block=partial(
            components.chain,
            block_args_pairs=[
                (unit_block, {'filters': filters})
                for filters in range(
                    params['feature_extract_init_filteres'],
                    params['feature_extract_final_filteres'],
                    params['feature_extract_filter_step'],
                )
            ],
        ),
        repetition=12,
        gather_func=lambda x: tf.concat(x, axis=-1)
    )

    output = components.net_in_net(
        features,
        [
            partial(unit_block, filters=24, kernel_size=1),
            partial(
                components.chain,
                block_args_pairs=[
                    (unit_block, {'filters': 8, 'kernel_size': 1}),
                    (unit_block, {'filters': 8}),
                ],
            )
        ],
        gather_func=lambda x: tf.concat(x, axis=-1),
    )

    output = unit_block(output, filters=32)
    output = tf.layers.conv2d_transpose(
        output, filters=16, kernel_size=4, strides=scale,
        activation=tf.nn.leaky_relu, use_bias=False,
    )
    output = tf.layers.conv2d(output, filters=1, kernel_size=3, activation=None, use_bias=False)

    bicubic = tf.image.resize_bicubic(lrimage, target_size)

    if params['add_bicubic']:
        output = output + bicubic

    predictions = {
        'predicted': output,
    }
    # Configure the Prediction Op (for PREDICT mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Estimators will save summaries while training session but not in eval or predict,
        #  so saver hook above is useful for eval and predict
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    hrimage = labels['hrimage']
    hrimage = tfops.image_central_crop_boundingbox(hrimage, target_size)

    loss = tf.losses.mean_squared_error(hrimage, output)

    tf.summary.image('predicted', predictions['predicted'])
    tf.summary.image('bicubic', bicubic)
    tf.summary.image('ground-truth', hrimage)
    tf.summary.image('input', lrimage)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[])

    # Add evaluation metrics (for EVAL mode)
    summary_saver_hook = tf.train.SummarySaverHook(
        save_steps=config.save_summary_steps,
        output_dir=os.path.join(config.model_dir, 'eval'),
        summary_op=tf.summary.merge_all())
    # Estimators will save summaries while training session but not in eval or predict,
    #  so saver hook above is useful for eval and predict
    eval_metric_ops = {
        'mse': tf.metrics.mean_squared_error(hrimage, output),
        'mae': tf.metrics.mean_absolute_error(hrimage, output),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[summary_saver_hook]
    )
