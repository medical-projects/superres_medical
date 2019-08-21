#!/usr/bin/python3
"""
this module will utilize upsample package
and train the model
"""
# built-in
from collections import OrderedDict

# external
import tensorflow as tf

# original
import upsample.engine as engine

candidates = OrderedDict([
    ('model', ['gan_circle_nongan']),
    ('add_bicubic', [True, False]),
    ('feature_extract_init_filteres', [32, 64]),
    ('feature_extract_final_filteres', [8, 16, 32]),
    ('feature_extract_filter_step', [3, 6]),
    ('upsample_scale', [2]),
    ('batch_size', [2]),
])


def main():
    engine.hyperparameter_optimize(
        datadir="/kw_resources/datasets/projects/upsample",
        candidates=candidates,
        output='/kw_resources/results/upsample/hyper_opt_res'
    )
    return


if __name__ == "__main__":
    main()
