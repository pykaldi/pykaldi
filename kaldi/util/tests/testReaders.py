from __future__ import division
import unittest
import numpy as np 

from kaldi.matrix import *
from kaldi.util import *

import os 

class _TestSequentialReaders:
    def test__init__(self):
        reader = self.getImpl()
        self.assertIsNotNone(reader)
        self.assertFalse(reader.IsOpen())
        self.assertFalse(reader.Done())

        # TODO (VM): What should this exception be?
        # Right now it is a RuntimeException in C++
        with self.assertRaises(Exception): 
            reader.Close()

        # Delete file in case it exists
        if os.path.exists('/tmp/temp.ark'):
            os.remove('/tmp/temp.ark')

        reader = self.getImpl('ark,t:/tmp/temp.ark')
        self.assertIsNotNone(reader)

        # Note (VM): If the file does not exist, this is false
        self.assertFalse(reader.IsOpen())

        # TODO (VM): This raises a runtime exception right now
        self.assertFalse(reader.Done())

        # Touch file
        open('/tmp/temp.ark', 'w').close()
        
        self.assertTrue(reader.IsOpen())
        self.assertTrue(reader.Done())

    def testContextManager(self):
        with self.getImpl() as reader:
            self.assertTrue(reader.IsOpen())
            self.assertFalse(reader.Done())
            
        self.assertFalse(reader.IsOpen())
        self.assertTrue(reader.Done())

        # Reset reader so that it doesnt by default pass the next ones
        reader = None

        # Touch file
        open('/tmp/temp.ark', 'w').close()
        
        with self.getImpl('ark,t:/tmp/temp.ark') as reader:
            self.assertTrue(reader.IsOpen())
            self.assertFalse(reader.Done())
            
        self.assertFalse(reader.IsOpen())
        self.assertTrue(reader.Done())

    def test__iter__(self):
        with self.getImpl() as reader:
            for k, v in reader:
                self.assertTrue(reader.IsOpen())
                self.assertFalse(reader.IsDone())

        self.assertFalse(reader.IsOpen())
        self.assertTrue(reader.Done())

        reader = None
        with self.getImpl('ark,t:/tmp/temp.ark') as reader:
            for k, v in reader:
                self.assertTrue(reader.IsOpen())
                self.assertFalse(reader.IsDone())

        self.assertFalse(reader.IsOpen())
        self.assertTrue(reader.Done()) 

    def testRead(self):
        # Create a file and write an example to it
        with open('/tmp/temp.ark', 'w') as outpt:
            self.writeExample(outpt)

        # Read back the example
        with self.getImpl('ark,t:/tmp/temp.ark') as reader:
            for idx, (k, v) in enumerate(reader):
                self.checkRead(idx, (k, v))

        self.assertFalse(reader.IsOpen())
        self.assertTrue(reader.Done())

class TestSequentialVectorReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialVectorReader(args[0])
        else:
            return SequentialVectorReader()

    def writeExample(self, outpt):
        pass

    def checkRead(self, idx, pair):
        self.fail("TODO")

class TestSequentialMatrixReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialMatrixReader(args[0])
        else:
            return SequentialMatrixReader()
    
    def writeExample(self, outpt):
            pass

    def checkRead(self, idx, pair):
        self.fail("TODO")


class TestSequentialFloatReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialFloatReader(args[0])
        else:
            return SequentialFloatReader()

    def writeExample(self, outpt):
        outpt.write("one 1.0\ntwo 2.0\nthree 3.0\n")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", 1.0), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", 2.0), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", 3.0), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialBoolReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialBoolReader(args[0])
        else:
            return SequentialBoolReader()

    def writeExample(self, outpt):
        outpt.write("one T\ntwo T\nthree F\n")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", True), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", True), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", False), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")
        
class TestSequentialIntVectorReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialIntVectorReader(args[0])
        else:
            return SequentialIntVectorReader()

    def writeExample(self, outpt):
        outpt.write("""one 1\n""" +\
                    """two 2 3\n""" +\
                    """three\n""")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [1]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [2, 3]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialIntVectorVectorReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialIntVectorVectorReader(args[0])
        else:
            return SequentialIntVectorVectorReader()

    def writeExample(self, outpt):
        outpt.write("""one 1 ;\n""" +\
                    """two 1 2 ; 3 4 ;\n""" +\
                    """three\n""")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [[1]]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [[1, 2], [3, 4]]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialIntPairVectorReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialIntPairVectorReader(args[0])
        else:
            return SequentialIntPairVectorReader()

    def writeExample(self, outpt):
        outpt.write("""one 1 1\n""" +\
                    """two 2 3 ; 4 5\n""" +\
                    """three\n""")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [(1, 1)]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [(2, 3), (4, 5)]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialFloatPairVectorReader(_TestSequentialReaders, unittest.TestCase):
    def getImpl(self, *args):
        if args:
            return SequentialFloatPairVectorReader(args[0])
        else:
            return SequentialFloatPairVectorReader()

    def writeExample(self, outpt):
        outpt.write("""one 1.0 1.0\n""" +\
                    """two 2.0 3.0 ; 4.0 5.0\n""" +\
                    """three\n""")

    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [(1.0, 1.0)]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [(2.0, 3.0), (4.0, 5.0)]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

if __name__ == '__main__':
    unittest.main()