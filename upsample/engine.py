'''
this module provides functionalities
to train/eval/tune the models

it also helps to deal with the saving and loading models
'''

# built-in
from functools import partial
from copy import deepcopy
from collections import OrderedDict
import sys
import os

# External
import skopt
import numpy as np
import tensorflow as tf

# Original
import upsample.utils as utils
import upsample.models as models
import upsample.data as data

logger = utils.loggings.get_standard_logger(__name__)

model_dir = "summary"
def get_estimator(
        model_dir=model_dir,
        save_interval=500,
        params=None,
        warm_start=None,
        model_specifier='dense_based',
        assign_gpu='auto',
        logger=logger,
):
    """
    this function returns Estimator
    Args:
        model_dir: (str) directory where chechpoints and summaries will be saved
        save_interval: interval in steps between saves
            You can manage the frequency of both summary and checkpoint
        params: parameters for model
            note that parameters will be automatically set either
            by default of by the config file under model_dir
            if params is None
            also, if you specify inproper parameter
        assign_gpu: 'auto' or int
            specifies the num of gpu assigned to this model
        model_specifier: str to specify the model
            if 'model' is specified in params,
            this param will be overwritten by it.
    """
    if assign_gpu == 'auto':
        assign_gpu = utils.resources.get_available_gpus()

    config_session = tf.ConfigProto(
        # inter_op_parallelism_threads=tio.FLAGS.cores,
        # intra_op_parallelism_threads=tio.FLAGS.cores,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    config = tf.estimator.RunConfig(
        train_distribute=tf.contrib.distribute.MirroredStrategy(
            num_gpus=assign_gpu,
        ),
        save_checkpoints_steps=save_interval,
        save_summary_steps=save_interval,
        session_config=config_session,
    )
    params_model = utils.param.load_config(model_dir, validate=False)
    params_warm = utils.param.load_config(warm_start, validate=False)
    params = utils.param.merge_dicts(params_warm, params_model, params)

    if params is not None:
        if 'model' in params:
            new_model_specifier = params['model']
            logger.warn('model_specifier overwritten: {} -> {}'.format(
                model_specifier, new_model_specifier,
            ))
            model_specifier = new_model_specifier
            print(model_specifier)
        else:
            if utils.param.config_validator(params, getattr(models, model_specifier).default_params):
                params['model'] = model_specifier
                logger.info('Assuming model type = {}'.format(model_specifier))
            else: RuntimeError('Please specify the model')

    model_module = getattr(models, model_specifier)
    print(model_module)
    print(model_specifier)
    params = utils.param.merge_params(params_warm, params_model, params, model_module.default_params)

    if params is not None:
        if model_dir is not None:
            utils.param.save_config(model_dir + "/config", params)

    print("Parameter: {}".format(dict(params)))
    if not utils.param.config_validator(params, model_module.default_params):
        RuntimeError('Invalid config detected')

    return tf.estimator.Estimator(
        model_fn=model_module.model,
        model_dir=model_dir,
        config=config,
        params=params,
        warm_start_from=warm_start,
    )


def instant_train(data_dir, model_dir, pretrained=None, steps=10000):
    """
    this function will train the model
    note that training conducted by this func
    is supposed to be just an instant one, not
    the intensive one.
    If you want to train the model seriously,
    use 'train' function instead.
    Args:
        tfrecord: str representing path for a tfrecord,
            or list of str, each of them representing tfrecord
        model_dir: (str) where to save the information of paramters and
            summaries
        pretrained: (str) path which holds checkpoint that is already trained
            this function will resume training from the checkpoint and
            save the new checkpoint in 'model_dir'
        steps: the number of steps to update this model

    Return:
        None
    """
    if steps is None: return
    print('INFO: starting instant_train for {} steps'.format(steps))
    dataset_provider = data.DatasetFactory(datadir=data_dir)

    estimator = get_estimator(
        model_dir=model_dir, save_interval=100, warm_start=pretrained
    )
    estimator.train(
        input_fn=dataset_provider.input_train(batch_size=estimator.param.batch_size),
        steps=steps,
    )
    return


def train(doc_name_train, doc_name_eval, label_name=None, params=None, iteration=1000, interval=1000, model_dir=None, model_dir_parent=None):
    """
    this function trains the model for adding label for each token
    Args:
        doc_name_train: name for a file which is either a pdf/txt or tfrecords
        doc_name_eval: name for a file which is either a pdf/txt or tfrecords
        label_name: name for CSV file. this can be 'None' if you specify
                    tfrecords files for a input file.
        params: (dict) paramters
        iteration: the number of train/eval
        interval: the steps of training in each iteration
        model_dir: (str) model directory
        model_dir_parent: (str) if this arg is provided, this func will decide the model_dir
            according to the params, and create the directory under 'model_dir_parent'
    """
    tf.summary.FileWriterCache.clear()
    if model_dir_parent is not None:
        model_dir = os.path.join(
            model_dir_parent,
            util.config_to_string(params, hash_=True, hash_table_file=os.path.join(model_dir_parent, 'hash_table'))
        )
    classifier = get_estimator(params=params, model_dir=model_dir, save_interval=interval)
    for i in range(iteration):
        classifier.train(
            input_fn=lambda: tio.input_func_train(doc_name_train, label_name),
            steps=interval,
        )
        eval_results = classifier.evaluate(
            input_fn=lambda: tio.input_func_test(doc_name_eval, label_name)
        )
    print(eval_results)


def hyperparameter_optimize(
        doc_name_train,
        doc_name_eval,
        candidates,
        output="hyper_opt_res",
        max_steps=5000,
        n_calls=200,
        allow_duplicate=False,
):
    """
    this function will perform hyperperameter optimization
    to the model and save the result to "output" file
    Since this function is extremly computationally expensive,
    we have decided to accept only tfrecords files as input data.
    That's the reason why this function does not accept 'label_file' as
    argument.

    Args:
        doc_name_train: the path for tfrecords files which to use for training
                        this can be either a str of a list of str
        doc_name_eval: the path for tfrecords files which to use for evaluation
                        this can be either a str of a list of str
        output: the output directory where to save the results
        max_steps: the max steps for each trial
        n_calls: the max num for trials
        allow_duplicate: when this is True, this function the trial will be
                skipped if it is already tried in the past.
        candidates: an instance of OrderedDict specifying
            parameters to try out together with its range

    This function will perform training for (max_steps * n_call) steps totally.
    """
    assert isinstance(candidates, OrderedDict), 'candidates must be OrderedDict'

    def wrapper(params_list, unfixed_params, fixed_params=[]):
        """wrapper func for get_estimator"""
        nonlocal counter
        params = determine_param(params_list, fixed_params=fixed_params, unfixed_params=unfixed_params)

        print("Trial {} / {}".format(counter, n_calls))
        print(dict(params))
        print()

        eval_res = None
        model_dir = os.path.join(
            output,
            util.config_to_string(
                params,
                hash_=True,
                hash_table_file=os.path.join(output, 'hash_table')),
        )
        config_path = os.path.join(model_dir, 'config_hyper_opt')
        prev_result = util.get_result(model_dir)

        if not allow_duplicate and prev_result is not None:
            print('INFO: Duplicate trial found, skippking...')
            eval_res = {'accuracy': prev_result}

        if eval_res is None:
            estimator = get_estimator(model_dir=model_dir, save_interval=500, params=params,)
            early_stop = tf.contrib.estimator.stop_if_no_decrease_hook(
                estimator=estimator,
                metric_name="loss",
                max_steps_without_decrease=estimator.config.save_checkpoints_steps * 7,
                run_every_secs=None,
                run_every_steps=estimator.config.save_checkpoints_steps,
            )
            try:
                eval_res, export_res = tf.estimator.train_and_evaluate(
                    estimator=estimator,
                    train_spec=tf.estimator.TrainSpec(
                        input_fn=lambda: tio.input_func_train(doc_name_train, None),
                        max_steps=max_steps, hooks=[early_stop],),
                    eval_spec=tf.estimator.EvalSpec(
                        input_fn=lambda: tio.input_func_test(doc_name_eval, None),
                        throttle_secs=0,),
                )
            except tf.train.NanLossDuringTrainingError:
                logger.error("Diverged")
                eval_res = {"accuracy": -1}
            except ValueError:
                e_type, e_value, e_traceback = sys.exc_info()
                logger.error('ValueError Occurred')
                traceback.print_exception(e_type, e_value, e_traceback)
                eval_res = {"accuracy": -2}

        counter += 1
        util.save_config(config_path, params)
        util.save_result(model_dir, eval_res['accuracy'])

        # Avoid the 'too many open files' issue.
        tf.summary.FileWriterCache.clear()
        return -eval_res["accuracy"]

    def determine_param(params_list, fixed_params, unfixed_params):
        '''
        using candidates, this function will determine and
        return the paramters.
        params_list is supposed to hold indices
        which indicates which value to take from
        each list contained in the candidate.
        '''
        params = dict()
        for index, param_idx in enumerate(params_list):
            key, val = list(candidates.items())[unfixed_params[index]]
            try: params[key] = val[param_idx]
            except IndexError:
                print(val)
                print(params_list)
                print(index)
                print(len(params_list))
                exit(0)
        for key, val in fixed_params:
            assert key not in params.keys(), 'Key:{} is conflicting'.format(key)
            params[key] = val
        return params

    def get_dimensions(dict_candidate):
        '''
        this function will return the list
        which can be accepted as a dimension by skopt API
        according to the specified dictionary
        '''
        dimensions = []
        unfixed_params = []
        fixed_params = []

        for idx, k_v in enumerate(dict_candidate.items()):
            k, v = k_v
            if len(v) == 1: fixed_params.append((k, v[0]))
            else:
                dimensions.append((0, len(v) - 1))
                unfixed_params.append(idx)
        return fixed_params, dimensions, unfixed_params

    counter = 0
    fixed_params, dimensions, unfixed_params = get_dimensions(candidates)

    res = skopt.gp_minimize(
        func=partial(wrapper, fixed_params=fixed_params, unfixed_params=unfixed_params),
        dimensions=dimensions,
        x0=[0] * len(dimensions),
        n_calls=n_calls,
    )

    with open(output + "whole_result", mode="w") as f:
        f.write(str(res))

    return res

def predict(file_path, model_dir=None, shuffle=False):
    """
    this function predicts the label for given input with
     weights that is generated from training session.
    """
    classifier = get_estimator(model_dir=model_dir)
    result = classifier.predict(
        input_fn=lambda: tio.input_func_predict(file_path, shuffle=shuffle)
    )
    return result
