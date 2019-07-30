"""
This module is a utility module named "tio" which
 stands for "Tensorflow Input / Output" and
 designed to support data preprocessing for tensorflow.
"""

import sys
import os
import multiprocessing
import regex
import tensorflow as tf
import utility as util

def create_simple_ds(data_dir, mode, no_healthy=False):
    '''
    this func will create simplest dataset
    which doesnt contain data augmentation or
    shuffling, etc...
    '''
    mode_error_msg = "Unexpected mode specified"

    if mode == "classification":
        dataset = create_simple_ds_for_classification(data_dir)
    elif mode == "annotation":
        dataset = create_simple_ds_for_annotation(data_dir, no_healthy=no_healthy)
    elif mode == "both":
        dataset = create_simple_ds_for_both(data_dir, no_healthy=no_healthy)
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)
        raise RuntimeError
    # at this point, dataset = dictionary
    return dataset

def decrement_group(dataset):
    '''
    this function will decrement group num in dataset
    '''
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            ('group', 'group', lambda x: x - 1),
            dictionary,
        )
    )
    return dataset

def determine_channel_size_on_mode(dataset, mode):
    '''
    this function will determine the channel size
    contained in ds depending on mode
    '''
    mode_error_msg = "Unexpected mode specified"
    if mode == "classification":
        dataset = determine_channel_size(dataset, 1)
    elif mode == "annotation" or mode == 'both':
        dataset = determine_channel_size(dataset, 2)
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)
        raise RuntimeError
    return dataset

def extract_from_dict(dict_in, keys_to_extract, remove):
    '''
    this function will extract elements from
    dict 'dict_in' and create new dict containing them
    '''
    assert isinstance(remove, bool)

    dict_out = dict()
    keys_to_extract = set(keys_to_extract)

    for key in keys_to_extract:
        dict_out[key] = dict_in[key]
        if remove:
            del dict_in[key]
    return dict_out

def separate_feature_label(dataset, mode):
    '''
    this func will separate features and labels
    in datasets
    '''
    mode_error_msg = "Unexpected mode specified"
    if mode == "classification":
        dataset = dataset.map(
            lambda dictionary: (
                dictionary,
                extract_from_dict(dictionary, ['group'], True)
            )
        )
    elif mode == "annotation":
        dataset = dataset.map(
            lambda dictionary: (
                dictionary,
                extract_from_dict(dictionary, ['annotation'], True)
            )
        )
    elif mode == "both":
        dataset = dataset.map(
            lambda dictionary: (
                dictionary,
                extract_from_dict(dictionary, ['annotation', 'group'], True)
            )
        )
    else:
        _print_error(mode_error_msg, sys._getframe().f_code.co_name)
        raise RuntimeError
    return dataset

def input_func_train(data_dir, mode, no_healthy=False):
    """
    Args:
        data_dir: directory that stores data
        mode: either "classification", "annotation", or "both"
        no_healthy: whether or not this func should remove
            healthy cases from data
    Returns:
        tf.Dataset = tuple(featuers<dict>, label)
    """
    dataset = create_simple_ds(data_dir, mode, no_healthy=no_healthy)
    dataset = determine_channel_size_on_mode(dataset, mode)
    dataset = augment_ds(dataset, mode)
    dataset = normalize(dataset)
    if mode == "annotation" or mode == 'both':
        dataset = divide_channels(dataset)
        # 'image' in dict will be divided into 'raw' 'annotation'

    dataset = separate_feature_label(dataset, mode)
    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch)
    dataset = dataset.repeat(None)
    return dataset

def modify(dataset, target_key, modify_rule_dict):
    '''
    this function will modify data with key 'target_key' in dataset
    regarding to dictionary 'modify_rule_dict'.
    modify_rule_dict is supposed to have keys,
    which represents domain and value represents codomain.
    note that modify_rule_dict must have all the values
    that is contained in values in key='target_key' in dataset
    as keys.
    '''
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            (target_key, target_key, lambda old: modify_rule_dict[old]),
            dictionary
        )
    )
    return dataset

def determine_channel_size(dataset, channel_size):
    '''
    this function will determines
    the channel size in the given dataset

    note that default channel_size is 2 because
    feature image and label image are concatenated
    '''
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            (
                'image', 'image',
                lambda image: util.tf_determine_image_channel(
                    image, channel_size)
            ), dictionary
        ),
    )
    return dataset


def divide_channels(dataset):
    """
    this function divides images which has 2 channel,
    which are supposed to be features channel and
    labels channel, into separate data with
    tuple object.

    Args:
        dataset: dataset consists of images with 2 channel
    Returns:
        dataset: dataset containing dict, which has 'raw' and 'annotation'
            as keys
    """
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            (
                ['image', 'raw', lambda raw_annotation: raw_annotation[:, :, :1]],
                ['image', 'annotation', lambda raw_annotation: raw_annotation[:, :, 1:]],
            ),
            dictionary),
    )
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            (
                ['image', None, None],  # delete image
            ),
            dictionary),
    )
    return dataset


def normalize(dataset):
    """
    this function map image
    [0,255] -> [0,1]
    """
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            (
                'image', 'image',
                lambda image: tf.div(tf.cast(image, tf.float32), 255.0),
            ),
            dictionary,
        ),
        FLAGS.cores,
    )
    return dataset


def create_simple_ds_for_annotation(data_dir, label_value=(255, 0, 0), no_healthy=False):
    """
    this function creates simple ds for segmentation

    Returns:
        dataset = (image, patient_id, img_id)
    """
    data_dir = data_dir if data_dir[-1] == "/" else data_dir + "/"

    if no_healthy:
        pattern = data_dir + "annotation/cancer_annotations/*/*"
    else:
        pattern = data_dir + "annotation/*/*/*"

    dataset = tf.data.Dataset.list_files(file_pattern=pattern, shuffle=True)
    # we need to add '*/*/*' at the end of data_dir
    #  because we suppose directory structure like below.
    #   data/train/annotation/cancer_annotations/5468464/23.jpg
    #   data/train/annotation/healthy_cases/5468464/23.jpg
    # NOTE: data_dir = "data/train/"

    # prepare the paths
    dataset = dataset.map(
        lambda path_annotation: (
            util.tf_get_raw(path_annotation),
            *tf_get_patient_img_from_path(path_annotation)),
        num_parallel_calls=FLAGS.cores,)
    # now dataset = (path_raw, path_anno, patient_id, image_id)

    # existence check
    dataset = dataset.filter(lambda raw, annotated,
                             *others: util.tf_exists([raw, annotated]))

    # load and decode
    dataset = dataset.map(
        lambda raw, annotated, *others: (
            tf.image.decode_image(tf.read_file(raw)),
            tf.image.decode_image(tf.read_file(annotated)),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # change the colorscale and extract labels
    dataset = dataset.map(
        lambda raw, annotated, *others: (
            tf.image.rgb_to_grayscale(raw),
            util.tf_extract_label(annotated, label_value),
            *others
        ),
        num_parallel_calls=FLAGS.cores,)

    # determine the size of the image
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            util.tf_determine_image_size(raw),
            util.tf_determine_image_size(annotation),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # crop
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            tf.image.resize_images(
                raw, [FLAGS.init_image_size, FLAGS.init_image_size]),
            tf.image.resize_images(
                annotation, [FLAGS.init_image_size, FLAGS.init_image_size]),
            *others,
        ),
        num_parallel_calls=FLAGS.cores,)

    # combine
    dataset = dataset.map(
        lambda raw, annotation, *others: (
            tf.concat([raw, annotation], 2),
            *others,
        ))
    # now dataset = (img_raw_annotation, patient_id, image_id)

    # dictionalize
    dataset = dataset.map(
        lambda combined_img, patient_id, img_id: {
            'image': combined_img,
            'patient_id': patient_id,
            'img_id': img_id,
        })
    return dataset


def query_group(patient_id, data_dir='./data', binarize=False):
    '''
    this func tries to figure out the cancer class
    for given patient id
    '''
    def __query(patient_id, data_dir):
        '''
        actual function
        needs to be wrapped
        '''
        if isinstance(data_dir, bytes):
            data_dir = data_dir.decode()
        if isinstance(patient_id, bytes):
            patient_id = patient_id.decode()
        group_dir = '{}/RAW/cancer_cases_grouped'.format(data_dir)
        if not isinstance(patient_id, str):
            patient_id = str(patient_id)

        for group in os.listdir(group_dir):
            if patient_id in os.listdir('{}/{}'.format(group_dir, group)):
                group_int = int(regex.sub(r'.*(\d+).*', r'\1', group))
                if binarize: group_int = 1 if group_int <= 3 else 2
                return group_int

        if patient_id in os.listdir('{}/RAW/healthy_cases'.format(data_dir)):
            return 0
        print('Error: failed to retrieve group for patient_id: {}'.format(patient_id))
        return -1
    return tf.py_func(__query, [patient_id, data_dir], tf.int64)

def create_simple_ds_for_both(data_dir, label_value=None, no_healthy=False, error_tolerant_mode=False):
    '''
    this function creates dataset that contatins
    (Raw MRI, Cancer Annotation, Cancer Stage Class)
    '''
    if label_value is None: label_value = util.label_value
    dataset = create_simple_ds_for_annotation(data_dir, label_value, no_healthy=no_healthy)
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict(
            ('patient_id', 'group', query_group), dictionary)
    )
    if no_healthy:
        dataset = decrement_group(dataset)
    if error_tolerant_mode:
        dataset = dataset.filter(
            lambda dictionary: tf.not_equal(dictionary['group'], -1))
    return dataset

def lambda_for_dict(src_dst_lambdas, target_dict):
    '''
    this func provides lambda functionatily
    which is especially useful when you
    want to apply some operation on
    the list of dictionaries, where
    you maybe want to apply some operation
    on just a part of the dictionary and
    want others to be the same.
    Args:
        src_dst_lambdas_lambda: list of source, destination and lambda
            source and destination must be elements of keys

            E.g.
            src_dst_lambdas_lambda = [(key0src, key0dst, lambda0), ...]
             -> d[key0dst] = lambda0(d[key0src])

            if destination is None, src will be removed
    '''
    def lambda_for_dict_single(src, dst, operation, target_dict):
        if dst is None:
            if src in target_dict.keys():
                del target_dict[src]
        else:
            target_dict[dst] = operation(target_dict[src])
        return target_dict

    if isinstance(src_dst_lambdas[0], str):
        src, dst, operation = src_dst_lambdas
        return lambda_for_dict_single(src, dst, operation, target_dict)
    else:
        for src, dst, operation in src_dst_lambdas:
            target_dict = lambda_for_dict_single(
                src, dst, operation, target_dict)
        return target_dict


def create_simple_ds_for_classification(data_dir):
    """
    this function creates simple ds for segmentation
    note that "create_simple_ds_for_classification"
    and "create_simple_ds_for_annotation" cannot be
    merged together due to the difference in directory
    structures.

    Returns:
        Dataset consits of dicts
        each dict has key[image, group, patient_id, img_id]
    """
    data_dir = data_dir if data_dir[-1] == "/" else data_dir + "/"

    dataset = tf.data.Dataset.list_files(
        file_pattern=data_dir + "classification/*/*/*", shuffle=True)
    # we need to add '*/*/*' at the end of data_dir
    #  because we suppose directory structure like below.
    #   data/train/Group1/5468464/23.jpg
    # NOTE: data_dir = "data/train/"

    dataset = dataset.map(tf_get_group_patient_img_from_path, FLAGS.cores)
    dataset = dataset.map(
        lambda image, group, patient_id, img_id: {
            'image': tf.image.decode_image(tf.read_file(image)),
            'group': group,  # Group Number
            'patient_id': patient_id,
            'img_id': img_id,
        },
    )
    dataset = dataset.map(
        lambda dictionary: lambda_for_dict((
            'image',
            'image',
            lambda image: tf.resize_images(
                tf.image.rgb_to_grayscale(util.tf_determine_image_size(image)),
                [FLAGS.init_image_size] * 2,
            ))),
        FLAGS.cores,
    )
    return dataset


def tf_get_group_patient_img_from_path(path):
    """wrapper func"""
    return tf.py_func(
        get_group_patient_img_from_path, [path], [
            tf.string, tf.int64, tf.int64, tf.int64]
    )


def tf_get_patient_img_from_path(path):
    """wrapper func"""
    return tf.py_func(
        get_patient_img_from_path, [
            path], [tf.string, tf.int64, tf.int64]
    )


def get_group_patient_img_from_path(path):
    """
    this function gets label for the image from path(str)

    images are supposed to have a path lile below
     'something/<group no.>/<patient ID>/index.<extension>'
    we can get the label from <group no.>

    Args:
        str path

    Return:
        (path, group, patient_id, img_id)
    """

    # TF converts strings into bytes object internally,
    #  so we have to convert them back to string.
    if isinstance(path, bytes):
        path = path.decode()

    group = regex.sub(
        r"^.*/Group(\d)/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\1", path)

    group = int(group)
    _, patient_id, img_id = get_patient_img_from_path(path)

    # group number starts from 1
    group -= 1
    return (path, group, patient_id, img_id)


def get_patient_img_from_path(path):
    """
    this function gets label for the image from path(str)

    images are supposed to have a path lile below
     'something/<patient ID>/index.<extension>'

    Args:
        str path

    Return:
        (path, patient_id, img_id)
    """

    # TF converts strings into bytes object internally,
    #  so we have to convert them back to string.
    if isinstance(path, bytes):
        path = path.decode()

    patient_id = regex.sub(
        r"^.*/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\1", path
    )
    img_id = regex.sub(
        r"^.*/(\d+)/(\d+)\.(jpg|jpeg|png|bmp)", r"\2", path)

    patient_id, img_id = list(map(int, [patient_id, img_id]))

    return (path, patient_id, img_id)


def augment_ds(ds, mode):
    """
    this function performs data augmentation
    to given dataset and returns dataset
    """

    def augment_static(image):
        """
        performs augmentation to given image
        """
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.5)
        return image

    def augment_dynamic(image):
        """
        performs augmentation by cropping/resizing
        given image
        """
        image = tf.image.resize_images(
            tf.image.crop_to_bounding_box(
                image,
                offset_height=tf.div(
                    FLAGS.init_image_size
                    - FLAGS.intermediate_image_size
                    + tf.random_uniform([], -10, 10, dtype=tf.int64),
                    2,
                ),
                offset_width=tf.div(
                    FLAGS.init_image_size
                    - FLAGS.intermediate_image_size
                    + tf.random_uniform([], -10, 10, dtype=tf.int64),
                    2,
                ),
                target_height=FLAGS.intermediate_image_size
                + tf.random_uniform([], -10, 10, dtype=tf.int64),
                target_width=FLAGS.intermediate_image_size
                + tf.random_uniform([], -10, 10, dtype=tf.int64),
            ),
            [FLAGS.final_image_size] * 2,
        )
        return image

    def augment_warp(image, n_points=100, width_index=0, height_index=1, threshold=5, default=0.0):
        '''
        this function will perfom data augmentation
        using Non-affine transformation, namely
        image warping.
        Currently, only square images are supported

        Args:
            image: input image
            n_points: the num of points to take for image warping
            width_index: index of width, set this to 1 if batched 0 otherwise normally
            height_index: index of height
            threshold: threshold to judge random value inappropriate
                diff values with its abs heigher width/threshold will be judged inappropriate
            default: value to be used when random value is over threshold
        Return:
            warped image
        '''
        width = int(image.get_shape()[width_index])
        height = int(image.get_shape()[height_index])
        assert width == height

        default = tf.cast(default, tf.float32)

        raw = tf.random_uniform([1, n_points, 2], 0.0, tf.cast(
            width, tf.float32), dtype=tf.float32)
        diff = tf.random_normal([1, n_points, 2], mean=0.0, dtype=tf.float32)
        # ensure that diff is not too big
        diff = default*tf.cast(tf.greater(tf.abs(diff), width/threshold), tf.float32) + \
            diff*tf.cast(tf.less_equal(tf.abs(diff),
                                       width/threshold), tf.float32)

        # expand dimension to meet the requirement of sparse_image_warp
        image = tf.expand_dims(image, 0)

        image = tf.contrib.image.sparse_image_warp(
            image=image,
            source_control_point_locations=raw,
            dest_control_point_locations=raw+diff,
        )[0]
        # sparse_image_warp function will return a tuple
        # (warped image, flow_field)

        # shrink dimension
        image = image[0, :, :, :]
        return image

    def fused_augmentation(image, augment_func):
        '''
        This func will perform data augmentation
        for images using functions supecified by
        augment_func

        Returned values are supposed to be sliced

        Note:
            initially, this func was implemented
            to return bunch of images
            that is augmentated,
            but we found that it is not good way
            to do the augmentation because
            similar data will be located close
            to each other and it will not sufficiently
            shuffled because the augmentation size
            is much more larger than shuffle buffer size.

            We finally found that the augmentation
            operation should receive one image and
            return one image that is changed a bit.
            In this way, we have various images
            through many epochs.

        Args:
            image: image data
            augment_func: function(s) used to augment
        Return:
            image
        '''
        if isinstance(augment_func, list) or isinstance(augment_func, tuple):
            for func in augment_func:
                image = func(image)
        else:
            image = augment_func(image)
        return image

    if mode == 'annotation':
        op_list = [augment_dynamic, augment_static, augment_warp]
    elif mode == 'warp':
        op_list = [augment_warp]
    else:
        op_list = [augment_dynamic, augment_static, ]
    print('INFO: augmentation:{}'.format(op_list))

    ds = ds.map(
        lambda dictionary: lambda_for_dict(
            (
                'image', 'image',
                lambda image: fused_augmentation(
                    image, op_list)
            ),
            dictionary
        ),
        num_parallel_calls=FLAGS.cores,)
    return ds


def resize_image(ds):
    """
    make images smaller so that they can be handled by DNN
    """
    ds = ds.map(
        lambda dictionary: lambda_for_dict(
            ('image', 'image',
             lambda image: tf.image.resize_images(
                 tf.image.crop_to_bounding_box(
                     image,
                     offset_height=tf.div(FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                     offset_width=tf.div(FLAGS.init_image_size - FLAGS.intermediate_image_size, 2),
                     target_height=FLAGS.intermediate_image_size,
                     target_width=FLAGS.intermediate_image_size,
                 ), [FLAGS.final_image_size] * 2,)
             ),
            dictionary
        ),
        FLAGS.cores,
    )
    return ds


def dataset_from_tfrecord(file_name):
    """
    this function generate dataset from tfrecord
    Args:
        file_name: file names for tfrecord
    Returns: dataset (unbatched)
    """
    file_name = [file_name] if not isinstance(file_name, list) else file_name
    file_num = len(file_name)

    dataset = tf.data.Dataset.from_tensor_slices(file_name).interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=1024 * 1024).map(
            lambda example: tf.parse_single_example(
                example,
                features={
                    "word": tf.FixedLenFeature([FLAGS.max_words], tf.string),
                    "label": tf.FixedLenFeature([FLAGS.max_words], tf.int64),
                },
            ),
            num_parallel_calls=FLAGS.cores,
        ),
        cycle_length=file_num,
        block_length=1,
    )
    dataset = dataset.map(
        lambda feature_dict: (
            {"word": feature_dict["word"]}, feature_dict["label"]),
        num_parallel_calls=FLAGS.cores,
    )
    return dataset


def is_tfrecord(doc_name):
    """this funciton judges if a file is a tfrecord"""
    if isinstance(doc_name, str):
        return doc_name.find(".tfrecords") != -1
    if isinstance(doc_name, list) and doc_name:
        return is_tfrecord(doc_name[0])
    return False


def input_func_test(data_dir, mode, no_healthy=False):
    """
    this function generates dataset for tensorflow Estimators
     leveraging dataset_from_doc function which is supposed to be
     base function for generating dataset.
    Args:
        doc_name: a file name for a document or a tfrecord file (must end with ".tfrecords")
        mode: mode 'annotation', 'classification', or 'both'
            annotation: raw MRI and annotation
            classification: raw MIR and label
            both: raw MRI and annotation and label
        no_healthy: whether this func should remove all the healthy cases
            from data or not
    Returns:
        tf.Dataset
            touple(dict, label)
    """
    dataset = create_simple_ds(data_dir, mode, no_healthy=no_healthy)
    dataset = determine_channel_size_on_mode(dataset, mode)
    dataset = resize_image(dataset)
    dataset = normalize(dataset)
    if mode == "annotation" or mode == 'both':
        dataset = divide_channels(dataset)
        # 'image' in dict will be divided into 'raw' 'annotation'

    dataset = separate_feature_label(dataset, mode)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(FLAGS.prefetch)
    return dataset


def input_func_predict(image_path):
    """
    this function converts string into dataset
    so that it can be passed to the model.
    """
    dataset = tf.data.Dataset.from_tensors(image_path)
    dataset = dataset.map(tf.read_file)
    dataset = dataset.map(tf.image.decode_image)
    dataset = resize_image(dataset)
    dataset = normalize(dataset)
    dataset = determine_channel_size(dataset)
    dataset = dataset.prefetch(FLAGS.prefetch)
    dataset = dataset.batch(1)
    return dataset


def generator_from_dataset(ds, sess):
    """
    this function creates generator object from dataset
    Args:
        dataset: dataset
        sess: tf.Session for evaluate tensors
    """
    iterator = ds.make_one_shot_iterator()
    next_row = iterator.get_next()
    try:
        while True:
            yield sess.run(next_row)
    except tf.errors.OutOfRangeError:
        pass


def generate_tf_record(path, ds, parse_fn=None):
    """
    this function generates tfrecords from dataset
    Args:
        path: direcotyr for tfrecords
        ds: dataset from which tfrecords are created
        parse_fn: parse function to generate example
    Returns:
        None
    """

    def _int64_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=make_sure_iterable(value)))

    def _bytes_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=make_sure_iterable(value)))

    def _float_feature(value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=make_sure_iterable(value)))

    def make_sure_iterable(value):
        if util.is_iterable(value):
            return value
        else:
            return [value]

    def row_to_example(row):
        """this function convert row to example"""
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "group": _int64_feature(row["group"]),
                    "patient_id": _int64_feature(row['patient_id']),
                    "image": _float_feature(row['image']),
                    "img_id": _int64_feature(row['img_id']),
                }
            )
        )

    if parse_fn is None:
        parse_fn = row_to_example

    config_session = tf.ConfigProto(
        inter_op_parallelism_threads=FLAGS.cores,
        intra_op_parallelism_threads=FLAGS.cores,
    )

    counter = 0
    with tf.Session(config=config_session) as sess, tf.python_io.TFRecordWriter(path) as writer:
        for row in generator_from_dataset(ds, sess):
            counter += 1
            print("[Creating TFRecord] NOW:" + str(counter))
            example = parse_fn(row)
            writer.write(example.SerializeToString())


class FLAGS:
    cores = multiprocessing.cpu_count()
    prefetch = None
    batch_size = 180
    init_image_size = 512
    intermediate_image_size = 180
    final_image_size = intermediate_image_size
    shuffle_buffer_size = 1 * batch_size

    gpus = len(util.get_available_gpus())
    if gpus != 0:
        batch_size = int(batch_size / gpus)
        print('INFO: batch size per GPU is {}\n'.format(batch_size))
