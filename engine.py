"""
module to leverage models
"""

import tensorflow as tf
import tio
import skopt
import utility as util
import os

def get_estimator(model_module, model_dir, save_interval=100, params=None, warm_start_setting=None):
    """
    this function returns Estimator
    Args:
        model_dir: (str) directory where chechpoints and summaries will be saved
    """
    model_fn = model_module.model_fn
    default_params = model_module.default_params

    config_session = tf.ConfigProto(
        inter_op_parallelism_threads=tio.FLAGS.cores,
        intra_op_parallelism_threads=tio.FLAGS.cores,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    config = tf.estimator.RunConfig(
        train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=tio.FLAGS.gpus),
        save_checkpoints_steps=save_interval,
        save_summary_steps=save_interval,
        session_config=config_session,
        model_dir=model_dir,
        keep_checkpoint_max=20,
    )
    if params is not None and not util.config_validator(params, default_params):
        print("WARGING: params are not valid. descarding...")
        params = None

    if params is None:
        params_temp = util.get_config_from_modeldir(model_dir)
        if params_temp is not None:
            print("INFO: config file found")
        if util.config_validator(params_temp, default_params):
            print("INFO: params in config file confirmed to be valid")
            params = params_temp
        else:
            print("INFO: params in config file is not valid. Ignored that config.")

    if params is not None and util.list_isin("batch_size", list(params.keys())):
        tio.FLAGS.batch_size = params['batch_size']

    return tf.estimator.Estimator(
        model_fn=model_fn, config=config, params=params, warm_start_from=warm_start_setting
    )


def train(model_module, data_dir, steps=3000, no_healthy=False, model_dir=None, warm_start_setting=None):
    """
    this function trains the model.
    Args:
        model_module: model module that contains
                model_fn and default_params
        steps: the num of steps to train
        mode: specify a mode 'classification' or 'annotation'
    Return:
        return value from evaluation of the model
    """
    estimator = get_estimator(model_module, model_dir=model_dir, warm_start_setting=warm_start_setting)
    eval_res, export_res = tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=lambda: tio.input_func_train(data_dir, no_healthy=no_healthy), max_steps=steps
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=lambda: tio.input_func_test(data_dir, no_healthy=no_healthy), throttle_secs=0
        ),
    )
    if eval_res is None:
        eval_res = estimator.evaluate(
            input_fn=lambda: tio.input_func_test(data_dir, True)
        )
    print(eval_res)
    return eval_res

def hyperparameter_optimize(model_module, data_dir, output="hyper_opt_res", max_steps=10000, n_calls=1000):
    """
    this function will perform hyperperameter optimization
    to the model and save the result to "output" file

    Args:
        model_module: model module
        mode: specify a mode 'classification' or 'annotation'
        output: the output directory where to save the results
        max_steps: the max steps for each trial
        n_calls: the max num for trials

    This function will perform training for (max_steps * n_call) steps totally.
    """
    output = output + "/" if output[-1] != "/" else output

    def wrapper(params_list):
        """wrapper func for get_estimator"""
        nonlocal counter
        params = {
            "unet_filters_first":params_list[0],
            "unet_n_downsample":params_list[1],
            "kernel_size":params_list[2],
            "conv_stride":params_list[3],
            "unet_rate":params_list[4],
        }

        print("Trial {} / {}".format(counter, n_calls))
        print('{}\n'.format(params))

        estimator = get_estimator(
            model_module,
            model_dir=output + util.config_to_file_name(params),
            save_interval=500,
            params=params,
        )
        early_stop = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name="loss",
            max_steps_without_decrease=estimator.config.save_checkpoints_steps * 8,
            run_every_secs=None,
            run_every_steps=estimator.config.save_checkpoints_steps,
        )

        try:
            eval_res, export_res = tf.estimator.train_and_evaluate(
                estimator=estimator,
                train_spec=tf.estimator.TrainSpec(
                    input_fn=lambda: tio.input_func_train(data_dir),
                    max_steps=max_steps,
                    hooks=[early_stop],
                ),
                eval_spec=tf.estimator.EvalSpec(
                    input_fn=lambda: tio.input_func_test(data_dir),
                    throttle_secs=0,
                ),
            )
            if eval_res is None:
                eval_res = estimator.evaluate(
                    input_fn=lambda: tio.input_func_test(data_dir)
                )
        except tf.train.NanLossDuringTrainingError:
            print("Diverged")
            eval_res = {"accuracy": -1}

        counter += 1

        with open(output + util.config_to_file_name(params) + "/result", mode="w") as f:
            f.write(str(eval_res["accuracy"]))

        return -eval_res["accuracy"]

    counter = 0
    res = skopt.gp_minimize(
        func=wrapper,
        dimensions=[ (50, 80), (3, 7), (3, 4), (1, 2), ],
        x0=[64, 5, 3, 1,],
        n_calls=n_calls,
    )

    with open(output + "result", mode="w") as f:
        f.write(str(res))

    return res

def predict(model_module, image):
    """
    this function classify a cancer
    Args:
        model_module: model module
        mode: specify a mode 'classification' or 'annotation'
        image: input image
    """
    classifier = get_estimator(model_module)
    result = classifier.predict(input_fn=lambda: tio.input_func_predict(image))
    return result
