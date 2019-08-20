'''
provides hash function with auto-saving feature
'''
# built-in
import pickle
import os

# external
from bidict import bidict

# original

class HashTable():
    def __init__(self, save_file=None):
        '''
        Instantiate a hastable class

        Args:
            save_file: a path to file where
                this file will read/write to
                store hash table
        '''
        self._hash_table = bidict()
        self._save_file = save_file

        self._load()
        return

    def get_hash(self, obj):
        '''
        get a hash value for a given object

        Args:
            obj: an object to hash
                if this arg is present,
                this func will return its hash_value
        '''
        obj = pickle.dumps(obj)
        if obj not in self._hash_table.keys():
            self._hash_table[obj] = len(self._hash_table)
            self._save()
        result = self._hash_table[obj]

        return result

    def get_obj(self, hash_value):
        '''
        get a obj corresponding to a given hash value

        Args:
            hash_value: a hash value of some object
                if this arg is present,
                this func will return its obj
        '''
        assert hash_value in self._hash_table.values()
        result = self._hash_table.inverse[hash_value]
        result = pickle.loads(result)
        return result

    def _load(self):
        if self._save_file is not None and os.path.exists(self._save_file):
            with open(self._save_file, 'rb') as f: self._hash_table = pickle.load(f)
        return

    def _save(self):
        save_dir = os.path.dirname(self._save_file)
        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        with open(self._save_file, 'wb') as f: pickle.dump(self._hash_table, f)
        return
