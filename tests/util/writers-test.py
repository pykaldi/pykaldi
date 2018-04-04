from __future__ import division
import os
import unittest

from kaldi.matrix import Vector, Matrix, SubMatrix, SubVector
from kaldi.util import *

from .mixins import *

# TODO (VM):
# This tests make use of the filesystem
# However at the moment I am assuming a
# Unix architecture (e.g., writing to /tmp).
# How to do a multi-platform setup?
class _TestWriters(AuxMixin):

    def test__init__(self):
        writer = self.getImpl() # call factory method
        self.assertIsNotNone(writer)
        self.assertFalse(writer.is_open())

        with self.assertRaises(Exception):
            writer.close()

        writer = self.getImpl(self.rspecifier)
        self.assertIsNotNone(writer)
        self.assertTrue(writer.is_open())
        self.assertTrue(writer.close())

        with self.assertRaises(RuntimeError):
            writer.close()

        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists(self.filename))

    def testContextManager(self):
        obj = self.getExampleObj()
        with self.getImpl(self.rspecifier) as writer:
            for o in obj:
                writer.write("myobj", o)

        # Check writer is closed
        self.assertFalse(writer.is_open())

        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists(self.filename))

    def test__setitem__(self):
        obj = self.getExampleObj()
        with self.getImpl(self.rspecifier) as writer:
            for o in obj:
                writer["myobj"] = o

        # Check writer is closed
        self.assertFalse(writer.is_open())

        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists(self.filename))

class TestVectorWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [Vector([1, 2, 3, 4, 5]),
                SubVector(Vector([1, 2, 3, 4, 5]))]

class TestMatrixWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [Matrix([[3, 5], [7, 11]]),
                SubMatrix(Matrix([[3, 5], [7, 11]]))]

class TestIntWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [3]

class TestFloatWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [5.0]

class TestBoolWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [True]

class TestIntVectorWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [[5, 6, 7, 8]]

class TestIntVectorVectorWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [[[5, 6, 7, 8], [9, 10, 11]]]

class TestIntPairVectorWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [[(1, 2), (3, 4), (5, 6)]]

class TestFloatPairVectorWriter(_TestWriters, unittest.TestCase):
    def getExampleObj(self):
        return [[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]]

if __name__ == '__main__':
    unittest.main()
