#!/usr/bin/python3
'''
this module will utilize upsample package
and train the model
'''
# built-in

# external
import tensorflow as tf

# original
import upsample.engine as engine


def main():
    engine.hyperparameter_optimize(
        datadir='/kw_resources',
        candidates=candidates,
    )
    return


if __name__ == '__main__':
    main()
