'''
test dataset
'''

# built in
import unittest
import random

# external
import tensorflow as tf
import cv2
import numpy as np

# original
from upsample.data import DatasetFactory

tf.enable_eager_execution()

class TestDataFeeder(unittest.TestCase):
    def SetUp(self):
        self.modes = ['train', 'eval']
        return

    def test_patches(self):
        patch_size = random.randrange(120, 300)
        df = DatasetFactory('', patch_size=patch_size)
        ds = df._dataset_patches({'path': '/home/yoshihiko/Downloads/test.jpg'}, 'output')
        for e in ds:
            patch = e['output'].numpy()
            self.assertEqual(patch.shape[0], patch_size)
            self.assertEqual(patch.shape[1], patch_size)
            cv2.imshow('test', (patch * 255).astype(np.uint8))
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'): break
        return


if __name__ == '__main__':
    unittest.main()
