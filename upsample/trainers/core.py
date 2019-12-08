from upsample import data
import argparse
from upsample.utils import arg_util
from upsample import engine


def main(args, model_params):
    df = data.DatasetFactory(**arg_util.pick_args(args, data.DatasetFactory))
    engine.train(
        dataset_provider=df,
        params=model_params,
        iteration=['iteration'],
        interval=args['interval'],
        model_dir=args['model_dir'],
    )
    return


def add_DataFactory_args(parser):
    arg_util.add_args(parser, data.DatasetFactory)
    return

def add_engine_args(parser):
    parser.add_argument('--iteration', default=100, type=int)
    parser.add_argument('--interval', default=100, type=int)
    parser.add_argument('--model_dir', default=None, type=str)
    return

def add_core_args(parser):
    add_DataFactory_args(parser)
    add_engine_args(parser)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_core_args(parser)
    main(vars(parser.parse_args()))
