import tables
import numpy as np

import collections


def save_dict_hdf5(d, file_name, compress=5, predictive=True, types={}):
    filters = tables.Filters(complib='zlib', complevel=compress)
    with tables.open_file(file_name, mode='w', filters=filters) as hdf5:
        for name, array in d.items():
            if name in types:
                converter = eval('%s_to_array' % types[name])
                array = converter(array)
            assert array.dtype.kind in ['i', 'f'], 'Only signed integers are supported'
            hdf5.create_carray('/', name, obj=encode_diff(array) if predictive else array)
        type_table_format = {'key': tables.StringCol(itemsize=30), 'serialize': tables.StringCol(itemsize=30)}

        hdf5.createTable('/', '__types__', type_table_format)
        if types:
            hdf5.root.__types__.append(types.items())


def load_dict_hdf5(file_name, predictive=True):
    with tables.open_file(file_name, mode='r') as hdf5:
        types = dict(hdf5.root.__types__.read())
        root_node = hdf5.get_node('/')
        fields = filter(lambda s: not s.startswith("_") and not s == '__types__', dir(root_node))
        d = dict([(field, np.array(getattr(root_node, field))) for field in fields])
    if predictive:
        for name, array in d.items():
            array = decode_diff(array)
            if name in types:
                converter = eval('array_to_%s' % types[name])
                d[name] = converter(array)
            else:
                d[name] = array
    return d


def encode_diff(a):
    shape = a.shape
    a_lin = a.ravel()
    a_lin_diff = np.hstack([a_lin[0], a_lin[1:] - a_lin[:-1]])
    return a_lin_diff.reshape(shape)


def decode_diff(a):
    shape = a.shape
    a_lin = np.cumsum(a.ravel())
    return a_lin.reshape(shape)


def counter_to_array(c):
    return np.vstack([np.array(c.keys()).T, c.values()]).astype('int')


def array_to_counter(a):
    return collections.Counter({tuple(key) if len(key) > 1 else key[0]: value
                                for key, value in zip(a[:-1, :].T, a[-1, :])})
