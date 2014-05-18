import tables
import numpy as np

import collections

import functools


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1])


def save_dict_hdf5(d, file_name, compress=5, types={}):
    filters = tables.Filters(complib='zlib', complevel=compress)
    with tables.open_file(file_name, mode='w', filters=filters) as hdf5:
        for name, array in d.items():
            if name in types:
                converter = eval('%s_to_array' % types[name])
                array = converter(array)
            assert array.dtype.kind in ['i', 'f'], 'Only signed integers are supported'
            hdf5.create_carray('/', name, obj=encode_diff(array))
        type_table_format = {'key': tables.StringCol(itemsize=50), 'serialize': tables.StringCol(itemsize=30)}

        hdf5.createTable('/', '__types__', type_table_format)
        if types:
            hdf5.root.__types__.append(types.items())


def load_dict_hdf5(file_name):
    hdf5 = tables.open_file(file_name, mode='r')

    types = dict(hdf5.root.__types__.read())
    fields = filter(lambda s: not s.startswith("_") and not s == '__types__', dir(hdf5.root))
    d = dict([(field, [functools.partial(lambda f, dummy: np.array(f), getattr(hdf5.root, field)), decode_diff]) for field in fields])
    for name, array in d.items():
        if name in types:
            d[name] += [eval('array_to_%s' % types[name])]

    for name, functors in d.items():
        d[name] = compose(*functors)

    return hdf5, d


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
    return collections.Counter(dict([(tuple(key) if len(key) > 1 else key[0], value)
                                     for key, value in zip(a[:-1, :].T, a[-1, :])]))


def get_list_part(l, piece, parts):
    n = len(l)

    step = n / float(parts)
    start = int(step * piece)
    end = int(step * (piece + 1))

    return l[start:end]
