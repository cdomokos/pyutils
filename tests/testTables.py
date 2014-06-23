#!/usr/bin/python


import unittest

import numpy as np

import pyutils.utils as utils

import collections

import os

np.random.seed(31337)


class testTables(unittest.TestCase):

    def setUp(self):

        self.a1 = np.arange(1000)
        self.a2 = np.arange(10000).reshape((100, 100))
        self.a3 = np.arange(1000).reshape((10, 10, 10))
        self.a4 = np.arange(1000).reshape((10, 10, 10)).astype('uint32')

        self.r1 = np.random.random(1000)
        self.r2 = np.random.random(10000).reshape((100, 100))
        self.r3 = np.random.random(1000).reshape((10, 10, 10))

        self.b1 = (np.random.random(11) > 0.5).astype('byte')
        self.b2 = (np.random.random(100001) > 0.5).astype('byte')
        self.b3 = (np.random.random(800) > 0.5).astype('byte')
        self.b4 = (np.random.random(80013) > 0.5).astype('byte')

        self.c1 = collections.Counter({1: 3, 3: 4, 5: 6, 123: 4243})
        self.c2 = collections.Counter({(1, 3): 3, (4, 3): 4, (5, 234): 6, (123, 0): 4243})
        self.c3 = collections.Counter({(1, 3, 3, 4): 3, (4, 3, 1, 2): 4, (5, 234, 3, 4): 6, (123, 0, 1, 2): 4243})

    def test_saveLoad(self):

        d = {"a1": self.a1, "a2": self.a2, "a3": self.a3, "a4": self.a4,
             "r1": self.r1, "r2": self.r2, "r3": self.r3,
             "b1": self.b1, "b2": self.b2, "b3": self.b3, "b4": self.b4,
             "c1": self.c1, "c2": self.c2, "c3": self.c3}

        utils.save_dict_hdf5(d, "tables_test.bin", types={"c1": "counter", "c2": "counter", "c3": "counter",
                                                          "b1": "bitmask", "b2": "bitmask", "b3": "bitmask", "b4": "bitmask"})

        f, d_load = utils.load_dict_hdf5("tables_test.bin")

        for key in d.keys():

            self.assertTrue(np.all(d[key] == d_load[key](0)))

        f.close()

        os.remove("tables_test.bin")


if __name__ == "__main__":
    unittest.main()
