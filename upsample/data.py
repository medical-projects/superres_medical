'''
provides functions for data io
'''
# built-in
import os
from functools import partial
from multiprocessing import cpu_count

# external
import tensorflow as tf

# original
from upsample.utils import tfops
import upsample.utils as utils

class DatasetFactory:
    def __init__(
            self,
            datadir,
            ncores='auto',
            ngpus='auto',
    ):
        self.datadir = datadir

        if ncores == 'auto': self.ncores = tf.data.experimental.AUTOTUNE
        else: self.ncores = ncores

        if ngpus == 'auto': self.ngpus = utils.resource.get_available_gpus()
        else: self.ngpus = ngpus
        return

    def input_train(
            self,
            downsample_method='bicubic',
            shuffle=True,
            shuffle_buffer=200,
            batch=True,
            batch_size=5,
            prefetch=True,
            repeat=True,
            prefetch_buffer='auto',
            size=(512, 512),
    ):
        if self.ngpus == 0:
            actual_batch_size = batch_size
        else:
            actual_batch_size = batch_size // self.ngpus

        if prefetch_buffer == 'auto':
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        dataset = self.dataset_list(mode='train')
        dataset = self.decode(dataset, size=size)
        dataset = self.add_downsampled(dataset, method=downsample_method)
        dataset = self._split_feature_label(dataset)
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(actual_batch_size)
        dataset = dataset.prefetch(prefetch_buffer)

        if repeat:
            dataset = dataset.repeat()

        return dataset

    def _split_feature_label(self, dataset):
        '''
        split label from features
        '''
        dataset = dataset.map(
            lambda x: tfops.dict_split(x, ['hrimage']),
            num_parallel_calls=self.ncores,
        )
        return dataset

    def input_eval(
            self,
            downsample_method='bicubic',
            batch=True,
            batch_size=5,
            prefetch=True,
            prefetch_buffer='auto',
            size=(512, 512),
    ):
        if prefetch_buffer == 'auto':
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        dataset = self.dataset_list(mode='eval')
        dataset = self.decode(dataset, size=size)
        dataset = self.make_patches(dataset, 'hrimage', 'hrimage')
        dataset = self.add_downsampled(dataset, method=downsample_method)
        dataset = self._split_feature_label(dataset)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def make_patches(
            self,
            dataset,
            source_key,
            target_key,
            remove_source=False,
            patch_size=64,
    ):
        '''
        extract patches out of image
        '''
        ksizes = [1, patch_size, patch_size, 1]
        strides = [1, patch_size // 2, patch_size // 2, 1]

        dataset = dataset.map(
            lambda x: tfops.dict_map(
                x, source_key, target_key,
                lambda image: tf.image.extract_image_patches(
                    image, ksizes=ksizes, strides=strides, rates=[1] * 4, padding='VALID',
                ),
            ),
            num_parallel_calls=self.ncores,
        )

        if remove_source:
            dataset = dataset.map(
                lambda x: tfops.dict_delete(x, [source_key]),
                num_parallel_calls=self.ncores,
            )
        return dataset

    def input_predict(
            self,
    ):
        pass

    def decode(self, dataset, tag='hrimage', normalize=True, size=None):
        dataset = dataset.map(
            lambda x: tfops.dict_map(x, 'path', 'hrimage', tfops.decode_image),
            num_parallel_calls=self.ncores,
        )

        def func(x):
            x = tf.subtract(tf.div(tf.cast(x, tf.float32), 255.0), 0.5)
            return x

        if normalize:
            dataset = dataset.map(
                lambda x: tfops.dict_map(x, 'hrimage', 'hrimage', func),
                num_parallel_calls=self.ncores,
            )

        if size is not None:
            dataset = dataset.map(
                lambda x: tfops.dict_map(
                    x, 'hrimage', 'hrimage', lambda image: tf.image.resize_images(image, size)
                ),
                num_parallel_calls=self.ncores,
            )

        return dataset

    def add_downsampled(self, dataset, tag='lrimage', method='bicubic', scale=0.5):
        '''
        add downsampled images to the dataset
        '''
        downsample = partial(tfops.scale_image, method=method, scale=scale)

        dataset = dataset.map(
            lambda x: tfops.dict_map(x, 'hrimage', 'lrimage', downsample),
            num_parallel_calls=self.ncores,
        )
        return dataset

    def dataset_list(
            self,
            mode,
            tag='path',
    ):
        '''
        this func will return a dataset consists of paths
        for the images

        each data entry in the dataset will be a dict
        each path will be stored with the specified tag into this dict
        '''
        assert mode in ('train', 'eval')
        pattern = os.path.join(self.datadir, mode, '*', '*', '*')
        dataset = tf.data.Dataset.list_files(pattern)
        dataset = dataset.map(lambda path: {tag: path}, self.ncores)
        return dataset
