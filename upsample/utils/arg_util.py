'''
provide convinient funcs for parsesing args
'''

# built-in
import inspect

# external

# original

def add_args(parser, func):
    '''
    add params of a func to a parser
    '''
    params = inspect.signature(func).parameters
    for key in params:
        default = params[key].default
        required = False
        if default is inspect.Parameter.empty:
            print(default)
            default = None
            required = True

        type_ = type(default) if default is not None else None
        parser.add_argument(
            '--{}'.format(key),
            default=default,
            type=type_,
            required=required,
            help='[{}] default: {}'.format(func.__name__, default)
        )
    return parser

def pick_args(args, func):
    '''
    pick args for a func from args
    which supposed to holds args for multiple funcs
    '''
    params = inspect.signature(func).parameters
    func_args = {k: v for k, v in args.items() if k in params}
    return func_args
