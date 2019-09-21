#!/usr/bin/python3
"""
this module will utilize upsample package
and train the model
"""
# built-in
from collections import OrderedDict
import argparse

# external

# original
import upsample.engine as engine
from upsample import data

candidates = OrderedDict([
    ('model', ['gan_circle_nongan']),
    ('add_bicubic', [True, False]),
    ('feature_extract_init_filteres', [32, 64]),
    ('feature_extract_final_filteres', [8, 16, 32]),
    ('feature_extract_filter_step', [-3, -6]),
    ('upsample_scale', [2]),
    ('batch_size', [2]),
])


def main(datadir, batch_size, output):
    dataset_provider = data.DatasetFactory(datadir=datadir, batch_size=batch_size)
    engine.hyperparameter_optimize(
        dataset_provider=dataset_provider,
        candidates=candidates,
        output=output,
        target_key='eval/mae',
        minimize=True,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, default='/kw_resources/datasets/projects/upsample')
    parser.add_argument('--output', default='/kw_resources/results/upsample/hyper_opt_res')
    parser.add_argument('--batch_size', default=10, type=int)
    args = parser.parse_args()
    main(**vars(args))
