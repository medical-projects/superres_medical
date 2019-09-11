'''
test dataset
'''

# built in

# external
import tensorflow as tf

# original
from upsample.data import DatasetFactory

def test_base(datadir):
    modes = ['train', 'eval']
    df = DatasetFactory(datadir)
    for mode in modes:
        ds = df.base(mode)
        print(ds)
        e = ds.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            while True:
                print(sess.run(e))
    return
