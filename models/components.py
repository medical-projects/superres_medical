'''
module which provides various components
'''

import tensorflow as tf

def densenet(input_, block, repetition, **block_args):
    '''
    densenet
    Args:
        block: constituent of densenet
        repetition: the num of blocks
        block_args: args for the block
    '''
    outputs = []
    next_input = None

    outputs.append(block(input_, **block_args))
    for _ in range(repetition):
        outputs.append(block(next_input, **block_args))
        next_input = tf.concat(outputs)

    return next_input

def semi_densenet(input_, block, repetition, **block_args):
    '''
    in this func, outputs from each block will be gathered
    and concated at the end.

    each layer will receive only from the closest previous layer.
    Args:
        block: constituent of densenet
        repetition: the num of blocks
        block_args: args for the block
    '''
    outputs = []

    next_input = input_
    for _ in range(repetition):
        outputs.append(block(next_input, **block_args))
        next_input = outputs[-1]

    output = tf.concat(outputs)
    return output

def net_in_net(input_, blocks, args_list=None):
    '''
    this func represents network in network structure

    if you specify N blocks as 'blocks' arg,
    then this func will parallelize N blocks

    Args:
        input_: input tensor
        blocks: list of blocks to be parallel
        args_list: (optional) args for each block
            if given, len(args_list) must be equal to
            len(blocks)
            each element must be dict
    '''
    outputs = []

    if not args_list:
        assert len(args_list) == len(blocks)
        for block, args in zip(blocks, args_list):
            outputs.append(blocks(input_, **args))
    else:
        for block in blocks:
            outputs.append(blocks(input_))

    return tf.concat(outputs)

def chain(input_, block_args_pairs):
    '''
    this func will chain a list of blocks
    withou any skip connections.

    Args:
        block_args_pairs: a list of tuple(block, args)
    '''
    output = input_
    for block, args in block_args_pairs:
        output = block(output, **args)
    return output
