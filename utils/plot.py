'''
plot module
supposed to provide functions to draw plots
'''
import random
from operator import add
import regex
import numpy as np
import urllib
import tensorflow as tf
from functools import reduce
from tensorflow.python.client import device_lib
import os
import multiprocessing as mp
import cv2
import wget
import pandas as pd
import matplotlib.pyplot as plt
label_font_size = 15

def plot_combined_eval_summary(csv_dir, save_dir):
    '''
    this func will plot combined figure for each metric
    of all the evaluations
    '''
    name_conversion = {
        'classifier_cheat_bn_proper': 4,
        'classifier_cheat': 2,
        'classifier_cheat_new': 3,
        'classifier': 1,
    }
    exclude_list = [
        'classifier_cheat_bn',
        'classifier_cheat_bn_smaller',
        'classifier_cheat_bn_warp',
    ]
    files = os.listdir(csv_dir)
    files = list(filter(lambda x: 'annotator' not in x, files))
    files = list(filter(lambda x: 'binary' not in x, files))
    files = list(map(lambda x: '{}/{}'.format(csv_dir, x), files))
    files = list(filter(lambda file_: 'Evaluation' in file_, files))

    tag_list = set(map(lambda file_: get_run_mode_tag(file_)[-1], files))

    for tag in tag_list:
        combine_targets = list(filter(lambda file_: get_run_mode_tag(file_)[-1] == tag, files))
        run_df_list = list(map(lambda file_: (get_run_mode_tag(file_)[0], pd.read_csv(file_, index_col=0)), combine_targets))
        series_list = []
        for run, df in run_df_list:
            series = df.iloc[:, 0].dropna()
            series.name = run
            series.name = regex.sub('summary_', '', series.name)
            if series.name in exclude_list:
                continue
            if series.name in name_conversion.keys():
                series.name = name_conversion[series.name]
            series_list.append(series)
        print('INFO: found {} evaluation series for {}'.format(len(series_list), tag))
        series_list = sorted(series_list, key=lambda x: x.name)
        for series in series_list:
            series.name = 'EXP{}'.format(series.name)

        if tag == 'accuracy':
            vars_ = list(map(lambda s: s.max(), series_list))
        elif tag == 'loss':
            vars_ = list(map(lambda s: s.min(), series_list))
        else:
            vars_ = list(map(lambda s: s.min(), series_list))
        index_list = np.arange(len(vars_)) * 2
        name_list = list(map(lambda s: '{}'.format(s.name, s.argmax()), series_list))
        # name_list = list(map(lambda s: '{}\n{}'.format(s.name, s.argmax()), series_list))
        _, ax = plt.subplots()
        plt.bar(index_list, vars_)
        plt.xticks(index_list, name_list)
        ax.set_ylabel(tag)
        ax.set_xlabel('configuration')
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.tight_layout()
        plt.savefig('{}/{}_bar.png'.format(save_dir, tag))
        plt.close()

        min_index = min(list(map(lambda s: s.index[-1], series_list)))
        series_list = list(map(lambda s: s.loc[:min_index], series_list))
        _, ax = plt.subplots()
        for series in series_list:
            series.plot(legend=True).set_ylabel(tag)
        plt.tight_layout()
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.savefig('{}/{}.png'.format(save_dir, tag))
        plt.close()

    return

def plot_summary_csv(csv_dir, save_dir, overwrite=True):
    '''
    this func will make figures for all the CSV files
    '''
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    files = os.listdir(csv_dir)
    files = list(filter(lambda x: '.csv' == x[-4:], files))

    for file_ in files:
        _, ax = plt.subplots()
        if os.path.exists('{}/{}'.format(save_dir, change_extension(file_, 'png'))):
            os.remove('{}/{}'.format(save_dir, change_extension(file_, 'png')))

        _, tag = get_run_tag(file_)
        ylabel = summary_tag_modifier(tag)
        df = pd.read_csv('{}/{}'.format(csv_dir, file_), index_col=0)
        if 'Evaluation' in df.columns and 'Training' in df.columns:
            print('INFO: plotting Eval and Train: {}'.format(file_))
            df.Evaluation.dropna().plot(legend=True).set_ylabel(ylabel)
            df.Training.dropna().plot(legend=True).set_ylabel(ylabel)
        else:
            df.iloc[:, 0].plot(legend=False).set_ylabel(ylabel)
        plt.tight_layout()
        ax.yaxis.label.set_size(label_font_size)
        ax.xaxis.label.set_size(label_font_size)
        plt.savefig('{}/{}'.format(save_dir, change_extension(file_, 'png')))
        plt.close()

