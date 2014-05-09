import tables
import numpy as np


def save_dict_hdf5(d, file_name, compress=5):
    filters = tables.Filters(complib='zlib', complevel=compress)
    with tables.open_file(file_name, mode='w', filters=filters) as hdf5:
        for name, array in d.items():
            hdf5.create_carray('/', name, obj=array)


def load_dict_hdf5(file_name):
    with tables.open_file(file_name, mode='r') as hdf5:
        root_node = hdf5.get_node('/')
        fields = filter(lambda s: not s.startswith("_"), dir(root_node))
        d = dict([(field, np.array(getattr(root_node, field))) for field in fields])
    return d
