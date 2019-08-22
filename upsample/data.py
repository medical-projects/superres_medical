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
            shuffle_buffer=5000,
            batch=True,
            batch_size=5,
            prefetch=True,
            repeat=True,
            prefetch_buffer='auto',
    ):
        if self.ngpus == 0:
            actual_batch_size = batch_size
        else:
            actual_batch_size = batch_size // self.ngpus

        if prefetch_buffer == 'auto':
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        dataset = self.dataset_list(mode='train')
        dataset = self.decode(dataset)
        dataset = self.add_downsampled(dataset, method=downsample_method)
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(actual_batch_size)
        dataset = dataset.prefetch(prefetch_buffer)

        if repeat:
            dataset = dataset.repeat()

        return dataset

    def input_eval(
            self,
            downsample_method='bicubic',
            batch=True,
            batch_size=5,
            prefetch=True,
            prefetch_buffer='auto',
    ):
        if prefetch_buffer == 'auto':
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        dataset = self.dataset_list(mode='eval')
        dataset = self.decode(dataset)
        dataset = self.add_downsampled(dataset, method=downsample_method)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def input_predict(
            self,
    ):
        pass

    def decode(self, dataset, tag='hrimage',):
        dataset = dataset.map(
            lambda x: tfops.dict_map(x, 'path', 'hrimage', tf.read_file),
            num_parallel_calls=self.ncores,
        )

        dataset = dataset.map(
            lambda x: tfops.dict_map(x, 'hrimage', 'hrimage', tfops.decode_image),
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
