'''
provides functions for data io
'''
# built-in
import os
from functools import partial
from multiprocessing import cpu_count
from pdb import set_trace

# external
import tensorflow as tf

# original
from upsample.utils import tfops
import upsample.utils as utils

class DatasetFactory:
    def __init__(
            self,
            datadir,
            size=(512, 512),
            patch_size=80,
            batch_size=5,
            prefetch=True,
            ncores='auto',
            ngpus='auto',
            downsample_scale=0.5,
            prefetch_buffer='auto',
            downsample_method='bicubic',
            shuffle=True,
            shuffle_buffer=1000,
            batch=True,
            channels=1,
    ):
        self.datadir = datadir
        self.size = size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.ncores = ncores
        self.ngpus = ngpus
        self.downsample_scale = downsample_scale
        self.prefetch_buffer = prefetch_buffer
        self.downsample_method = downsample_method
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.batch = batch
        self.channels = channels

        if ncores == 'auto': self.ncores = tf.data.experimental.AUTOTUNE
        if ngpus == 'auto': self.ngpus = utils.resource.get_available_gpus()
        if prefetch_buffer == 'auto': self.prefetch_buffer = tf.data.experimental.AUTOTUNE
        return

    def train(self, repeat=True,):
        if self.ngpus == 0: actual_batch_size = self.batch_size
        else: actual_batch_size = self.batch_size // self.ngpus

        dataset = self.base(mode='train')
        dataset = dataset.batch(actual_batch_size)
        if self.prefetch: dataset = dataset.prefetch(self.prefetch_buffer)
        if repeat: dataset = dataset.repeat(None)
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

    def base(self, mode,):
        '''
        generate base dataset
        '''
        dataset = self.dataset_list(mode=mode)
        dataset = self.decode(dataset)
        dataset = self.add_downsampled(
            dataset,
            method=self.downsample_method,
            scale=self.downsample_scale,
        )
        dataset = self._split_feature_label(dataset)
        if self.shuffle: dataset = dataset.shuffle(self.shuffle_buffer)
        return dataset

    def eval(
            self,
            downsample_method='bicubic',
            batch=True,
    ):
        dataset = self.base(mode='eval')
        dataset = dataset.batch(self.batch_size)
        if self.prefetch: dataset = dataset.prefetch(self.prefetch_buffer)
        return dataset

    def input_predict(
            self,
    ):
        pass

    def _dataset_patches(self, dict_, store_key, preserve_input=True):
        '''
        make a dataset containing small patches

        Args:
            dict_: dictionary data
                this dict should have key:"path"
                from which patches are taken.
                all the data in this dict will be preserved
            store_key: to which key to save patches
        '''
        ksizes = [1, self.patch_size, self.patch_size, 1]
        strides = [1, self.patch_size // 2, self.patch_size // 2, 1]
        image = tfops.decode_image(dict_['path'], channels=self.channels)
        nchannels = image.get_shape()[-1]
        image = tf.div(tf.cast(image, tf.float32), 255.0)
        image = tf.expand_dims(image, 0)
        patches = tf.image.extract_image_patches(
            image, ksizes=ksizes, strides=strides, rates=[1] * 4, padding='VALID',
        )
        npixels = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [-1, npixels])
        patches = tf.reshape(patches, [-1, *ksizes[1:3], nchannels])
        dataset = tf.data.Dataset.from_tensor_slices(patches)

        if preserve_input:
            dataset = dataset.map(
                lambda patch: tfops.dict_add(dict_, store_key, patch),
                num_parallel_calls=self.ncores,
            )
        return dataset

    def decode(self, dataset, tag='hrimage', normalize=True, size=None):
        dataset = dataset.interleave(
            lambda x: self._dataset_patches(x, 'hrimage'),
            cycle_length=10,
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
