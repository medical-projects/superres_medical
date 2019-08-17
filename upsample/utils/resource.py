'''
various functions for managing hardware resources
'''
from tensorflow.python.client import device_lib

def get_available_gpus(return_list=False):
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if return_list:
        return gpu_list
    else: return len(gpu_list)
