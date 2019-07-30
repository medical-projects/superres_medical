#!/usr/bin/python

import model_classifier
import engine
import tensorflow as tf

def main():
    '''main'''
    engine.train(
        model_classifier, 'both', steps=10000000, no_healthy=True,
        model_dir='summary/summary_classifier',
    )
    return

if __name__ == '__main__':
    main()
