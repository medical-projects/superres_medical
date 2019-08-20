'''
various functions to deal with results
'''

# built-in
import pickle
import os

# external

# original

def save(result, save_dir, file_name='result.pickle'):
    '''
    save the result
    '''
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(result, f)
    return

def load(save_dir, file_name='result.pickle'):
    '''
    load a result
    '''
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'rb') as f:
        result = pickle.load(f)
    return result

def exists(save_dir, file_name='result.pickle'):
    '''
    checks if the result is present
    '''
    file_path = os.path.join(save_dir, file_name)
    return os.path.exists(file_path)
