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
        next_input = tf.stack(outputs)

    return next_input

def semi_densenet(
        input_,
        block,
        repetition,
        gather_func=tf.stack,
        stack_in_channel=True,
):
    '''
    in this func, outputs from each block will be gathered
    and concated at the end.

    each layer will receive only from the closest previous layer.
    Args:
        block: constituent of densenet
        repetition: the num of blocks
        block_args: args for the block
        gather_func: the function to gather results
            this func is supposed to receive a list of tensors
            and return one tensor
        stack_in_channel: whether or not the results should be stacked
            in the channel axis(axis=2).
            this is recommended if your data is image.
    '''
    outputs = []

    next_input = input_
    for _ in range(repetition):
        print('HERE')
        outputs.append(block(next_input))
        next_input = outputs[-1]

    output = gather_func(outputs)
    exit(0)
    return output

def net_in_net(
        input_,
        blocks,
        gather_func=tf.stack,
        args_list=None,
):
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

    if args_list:
        assert len(args_list) == len(blocks)
        for block, args in zip(blocks, args_list):
            outputs.append(block(input_, **args))
    else:
        for block in blocks:
            outputs.append(block(input_))

    return gather_func(outputs)

def chain(input_, block_args_pairs):
    '''
    this func will chain a list of blocks
    withou any skip connections.

    Args:
        block_args_pairs: a list of tuple(block, args)
    '''
    output = input_
    print(block_args_pairs)
    for block, args in block_args_pairs:
        print('chain')
        output = block(output, **args)
    return output
