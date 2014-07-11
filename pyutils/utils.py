import tables
import numpy as np

import collections

import functools

import os


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1])


def append_hdf5(hdf5, key, value, value_type=None):
    if value_type:
        converter = eval('%s_to_array' % value_type)
        value = converter(value)
    if value.dtype.kind in ['i', 'f']:
        hdf5.create_carray('/', key, obj=encode_diff(value))
    else:
        hdf5.create_carray('/', key, obj=value)

    if value_type:
        hdf5.root.__types__.append([(key, value_type)])

    return 0


def save_dict_hdf5(d, file_name, compress=5, types={}):
    filters = tables.Filters(complib='zlib', complevel=compress)
    with tables.open_file(file_name, mode='w', filters=filters) as hdf5:
        type_table_format = {'key': tables.StringCol(itemsize=50), 'serialize': tables.StringCol(itemsize=30)}
        hdf5.create_table('/', '__types__', type_table_format)

        for name, array in d.items():

            value_type = types[name] if name in types else None

            append_hdf5(hdf5, name, array, value_type)
    return 0


def load_dict_hdf5(file_name):
    hdf5 = tables.open_file(file_name, mode='r')

    types = dict(hdf5.root.__types__.read())
    fields = filter(lambda s: not s.startswith("_") and not s == '__types__', dir(hdf5.root))
    d = dict([(field, [functools.partial(lambda f, dummy: np.array(f), getattr(hdf5.root, field))]) for field in fields])
    for name, array in d.items():
        if getattr(hdf5.root, name).dtype.kind in ['i', 'f']:
            d[name] += [decode_diff]
        if name in types:
            d[name] += [eval('array_to_%s' % types[name])]

    for name, functors in d.items():
        d[name] = compose(*functors)

    return hdf5, d


def assemble_matrix_from_folder(folder_name, matrix_names, glue_functions):

    matrices = collections.defaultdict(list)

    for file_name in sorted(os.listdir(folder_name)):

        file_name = os.path.join(folder_name, file_name)

        hdf5, d = load_dict_hdf5(file_name)

        for matrix_name in matrix_names:
            matrices[matrix_name].append(d[matrix_name](0))

        hdf5.close()

    for key, glue in zip(matrix_names, glue_functions):
        value = matrices[key]
        matrices[key] = glue(value)

    return matrices


def assemble_matrix_hdf(file_name, matrix_prefixes, lbound=None, rbound=None):
    hdf5, d = load_dict_hdf5(file_name)

    matrices = []
    for prefix in matrix_prefixes:
        matrix_keys = sorted(filter(lambda s: s.startswith(prefix), d.keys()))
        matrix = [d[key](0) for key in matrix_keys[lbound:rbound]]
        matrices.append(matrix)

    hdf5.close()

    return matrices


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


def int_to_array(i):
    return np.array([i]).astype('int')


def array_to_int(a):
    return a[0]


def array_to_array(a):
    return a.astype('int')


def array_to_barray(a):
    return a.astype('byte')


def barray_to_array(a):
    return a.astype('int')


def bitmask_to_array(b):
    n = len(b)
    remove = [0] * 8
    remove[-n % 8] = 1
    b = np.hstack([remove, b])
    return np.packbits(b)


def array_to_bitmask(b):
    remove = np.unpackbits(b[0])
    cut = np.argmax(remove)
    return np.unpackbits(b)[8:-cut if cut else None]


def get_list_part(l, piece, parts):
    n = len(l)

    step = n / float(parts)
    start = int(step * piece)
    end = int(step * (piece + 1))

    return l[start:end]


def split_array(l, counts):

    i = 0
    l_splits = []
    for c in counts:
        l_splits.append(l[i:i + c])
        i += c
    assert(i == np.sum(counts))
    return l_splits
