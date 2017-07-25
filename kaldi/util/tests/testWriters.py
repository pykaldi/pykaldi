from __future__ import division
import unittest
import numpy as np 

from kaldi.matrix import *
from kaldi.util import *

import os 

# TODO (VM):
# This tests make use of the filesystem
# However at the moment I am assuming a 
# Unix architecture (e.g., writing to /tmp).
# How to do a multi-platform setup?
class _TestWriters:

    def test__init__(self):
        writer = self.getImpl() # call factory method
        self.assertIsNotNone(writer)
        self.assertFalse(writer.IsOpen())

        # TODO (VM): What should this exception be?
        # Right now it is a RuntimeException in C++
        with self.assertRaises(Exception): 
            writer.Close()

        writer = self.getImpl("ark:/tmp/temp.ark")
        self.assertIsNotNone(writer)
        self.assertTrue(writer.IsOpen())
        self.assertTrue(writer.Close())

        # TODO (VM): What should this exception be?
        # Right now it is a RuntimeException in C++
        with self.assertRaises(Exception): 
            writer.Close()
        
        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists("/tmp/temp.ark"))
        
    def testContextManager(self):
        obj = self.getExampleObj()

        with self.getImpl("ark,t:/tmp/temp.ark") as writer:
            writer.Write("myobj", obj)

        # Check writer is closed
        self.assertFalse(writer.IsOpen())

        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists("/tmp/temp.ark"))

        # Check the contents of the file
        # TODO (VM)...

    def test__setitem__(self):
        obj = self.getExampleObj()
        with self.getImpl("ark,t:/tmp/temp.ark") as writer:
            writer["myobj"] = obj

        # Check writer is closed
        self.assertFalse(writer.IsOpen())

        # Check that the file exists after closing the writer
        self.assertTrue(os.path.exists("/tmp/temp.ark"))

        # Check the contents of the file
        # TODO (VM)...

    def tearDown(self):
        if os.path.exists("/tmp/temp.ark"):
            os.remove("/tmp/temp.ark")

class TestVectorWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return VectorWriter(args[0])
        else:
            return VectorWriter()

    def getExampleObj(self):
        return Vector.new([1, 2, 3, 4, 5])

class TestMatrixWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return MatrixWriter(args[0])
        else:
            return MatrixWriter()

    def getExampleObj(self):
        return Matrix.new([[3, 5], [7, 11]])

class TestIntWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return IntWriter(args[0])
        else:
            return IntWriter()

    def getExampleObj(self):
        return 3

class TestFloatWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return FloatWriter(args[0])
        else:
            return FloatWriter()

    def getExampleObj(self):
        return 5.0

class TestBoolWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return BoolWriter(args[0])
        else:
            return BoolWriter()

    def getExampleObj(self):
        return True

class TestIntVectorWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return IntVectorWriter(args[0])
        else:
            return IntVectorWriter()

    def getExampleObj(self):
        return [5, 6, 7, 8]

class TestIntVectorVectorWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return IntVectorVectorWriter(args[0])
        else:
            return IntVectorVectorWriter()

    def getExampleObj(self):
        return [[5, 6, 7, 8], [9, 10, 11]]

class TestIntPairVectorWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return IntPairVectorWriter(args[0])
        else:
            return IntPairVectorWriter()
    def getExampleObj(self):
        return [(1, 2), (3, 4), (5, 6)]

class TestFloatPairVectorWriter(_TestWriters, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return FloatPairVectorWriter(args[0])
        else:
            return FloatPairVectorWriter()
            
    def getExampleObj(self):
        return [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


if __name__ == '__main__':
    unittest.main()
