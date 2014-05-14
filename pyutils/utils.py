import tables
import numpy as np


def save_dict_hdf5(d, file_name, compress=5, predictive=True):
    filters = tables.Filters(complib='zlib', complevel=compress)
    with tables.open_file(file_name, mode='w', filters=filters) as hdf5:
        for name, array in d.items():
            hdf5.create_carray('/', name, obj=encode_diff(array) if predictive else array)


def load_dict_hdf5(file_name, predictive=True):
    with tables.open_file(file_name, mode='r') as hdf5:
        root_node = hdf5.get_node('/')
        fields = filter(lambda s: not s.startswith("_"), dir(root_node))
        d = dict([(field, np.array(getattr(root_node, field))) for field in fields])
    if predictive:
        for name, array in d.items():
            d[name] = decode_diff(array)
    return d


def encode_diff(a):
    shape = a.shape
    a_lin = a.ravel()
    a_lin_diff = np.hstack([a_lin[0], np.diff(a_lin)])
    return a_lin_diff.reshape(shape)


def decode_diff(a):
    shape = a.shape
    a_lin = np.cumsum(a.ravel())
    print(shape)
    print(a_lin)
    return a_lin.reshape(shape)
