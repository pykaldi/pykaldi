from __future__ import division
import numpy as np
import os
import unittest

from kaldi.matrix import *
import kaldi.util

from .mixins import *

################################################################################################################
# Sequential Readers
################################################################################################################
class _TestSequentialReaders(AuxMixin):
    def test__init__(self):

        # Empty reader
        reader = self.getImpl()
        self.assertIsNotNone(reader)
        with self.assertRaises(RuntimeError):
            self.assertFalse(reader.is_open())
            self.assertFalse(reader.done())

        with self.assertRaises(RuntimeError):
            reader.close()

        # Delete file in case it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

        # Reader into a file that does not exists
        with self.assertRaises(IOError):
            reader = self.getImpl(self.rspecifier)

        # Touch file
        open(self.filename, 'w').close()

        reader = self.getImpl(self.rspecifier)
        self.assertTrue(reader.is_open())
        self.assertTrue(reader.done())

    def testContextManager(self):
        # Delete file in case it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

        # Empty reader via CM
        with self.assertRaises(RuntimeError):
            with self.getImpl() as reader:
                self.assertFalse(reader.is_open())
                self.assertFalse(reader.done())

            self.assertFalse(reader.is_open())
            self.assertFalse(reader.done())

        # Reset reader so that it doesnt by default pass the next ones
        reader = None

        # Touch file
        open(self.filename, 'w').close()

        with self.getImpl(self.rspecifier) as reader:
            self.assertTrue(reader.is_open())
            self.assertTrue(reader.done())

        with self.assertRaises(RuntimeError):
            self.assertFalse(reader.is_open())
            self.assertTrue(reader.done())

    def test__iter__(self):
        # Create a file and write an example to it
        with open(self.filename, 'w') as outpt:
            self.writeExample(outpt)

        # Iterate over the file
        with self.getImpl(self.rspecifier) as reader:
            for idx, (k, v) in enumerate(reader):
                self.checkRead(idx, (k, v))

        # Check iteration is done
        # FIXME:
        # This raises C++ exception instead of a simple True
        # self.assertTrue(reader.done())

        # Check iterator is closed
        self.assertFalse(reader.is_open())

class TestSequentialVectorReader(_TestSequentialReaders, unittest.TestCase, VectorExampleMixin):
    def checkRead(self, idx, pair):
        k, v = pair
        if idx == 0:
            self.assertEqual("one", k)
            self.assertTrue(np.array_equal([3.0, 5.0, 7.0], v.numpy()))
        elif idx == 1:
            self.assertEqual("two", k)
            self.assertTrue(np.array_equal([1.0, 2.0, 3.0], v.numpy()))
        elif idx == 2:
            self.assertEqual("three", k)
            self.assertEqual(0, len(v.numpy()))
        else:
            self.fail("shouldn't happen")

class TestSequentialMatrixReader(_TestSequentialReaders, unittest.TestCase, MatrixExampleMixin):
    def checkRead(self, idx, pair):
        k, m = pair
        if idx == 0:
            self.assertEqual("one", k)
            self.assertTrue(np.array_equal(np.arange(9).reshape((3, 3)), m.numpy()))
        elif idx == 1:
            self.assertEqual("two", k)
            self.assertTrue(np.array_equal([[1.0], [2.0], [3.0]], m.numpy()))
        elif idx == 2:
            self.assertEqual("three", k)
            self.assertEqual(0, len(m.numpy()))
        else:
            self.fail("shouldn't happen")

class TestSequentialWaveReader(_TestSequentialReaders, unittest.TestCase, WaveExampleMixin):
    def checkRead(self, idx, pair):
        k, m = pair
        if idx == 0:
            self.assertTrue("one", k)
            self.assertTrue(np.array_equal(np.arange(9).reshape((3, 3)), m.data().numpy()))
        else:
            self.fail("shouldn't happen")

class TestSequentialFloatReader(_TestSequentialReaders, unittest.TestCase, FloatExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", 1.0), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", 2.0), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", 3.0), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialBoolReader(_TestSequentialReaders, unittest.TestCase, BoolExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", True), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", True), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", False), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialIntVectorReader(_TestSequentialReaders, unittest.TestCase, IntVectorExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [1]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [2, 3]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialIntVectorVectorReader(_TestSequentialReaders, unittest.TestCase, IntVectorVectorExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [[1]]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [[1, 2], [3, 4]]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialIntPairVectorReader(_TestSequentialReaders, unittest.TestCase, IntPairVectorExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [(1, 1)]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [(2, 3), (4, 5)]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

class TestSequentialFloatPairVectorReader(_TestSequentialReaders, unittest.TestCase, FloatPairVectorExampleMixin):
    def checkRead(self, idx, pair):
        if idx == 0:
            self.assertTupleEqual(("one", [(1.0, 1.0)]), pair)
        elif idx == 1:
            self.assertTupleEqual(("two", [(2.0, 3.0), (4.0, 5.0)]), pair)
        elif idx == 2:
            self.assertTupleEqual(("three", []), pair)
        elif idx < 0 or idx > 2:
            self.fail("shouldn't happen")

################################################################################################################
# Random Access Readers
################################################################################################################
class _TestRandomAccessReaders(AuxMixin):

    def test__init__(self):
        reader = self.getImpl()
        self.assertIsNotNone(reader)
        self.assertFalse(reader.is_open())

        with self.assertRaises(RuntimeError):
            reader.close()

        # Delete file in case it exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

        # Reading a non-existant file raises an exception
        with self.assertRaises(IOError):
            reader = self.getImpl(self.rspecifier)
            self.assertIsNotNone(reader)

        # Note (VM): If the file does not exist, this is false
        self.assertFalse(reader.is_open())

        # Touch file
        open(self.filename, 'w').close()

        reader = self.getImpl(self.rspecifier)
        self.assertTrue(reader.is_open())

    def testContextManager(self):
        with self.assertRaises(RuntimeError):
            with self.getImpl() as reader:
                self.assertFalse(reader.is_open())

            self.assertFalse(reader.is_open())

        # Reset reader so that it doesnt by default pass the next ones
        reader = None

        # Touch file
        open(self.filename, 'w').close()

        with self.getImpl(self.rspecifier) as reader:
            self.assertTrue(reader.is_open())

        self.assertFalse(reader.is_open())

    def getValidKey(self):
        return "one"

    def getNotValidKey(self):
        return "four"

    def test__contains__(self):
        # Create a file and write an example to it
        with open(self.filename, 'w') as outpt:
            self.writeExample(outpt)

        # Empty impl
        with self.assertRaises(RuntimeError):
            self.assertFalse(self.getImpl().__contains__(self.getValidKey()))
            self.assertFalse(self.getImpl().__contains__(self.getNotValidKey()))

        # Check keys in example
        self.assertTrue(self.getImpl(self.rspecifier).__contains__(self.getValidKey()))
        self.assertFalse(self.getImpl(self.rspecifier).__contains__(self.getNotValidKey()))

        # Also check for the *in* operator
        self.assertTrue(self.getValidKey() in self.getImpl(self.rspecifier))
        self.assertFalse(self.getNotValidKey() in self.getImpl(self.rspecifier))

    def test__getitem__(self):
        # Create a file and write an example to it
        with open(self.filename, 'w') as outpt:
            self.writeExample(outpt)

        #
        self.checkRead(self.getImpl(self.rspecifier))

        # Check that the keys are strings
        with self.assertRaises(TypeError):
            self.getImpl(self.rspecifier)[1]

class TestRandomAccessVectorReader(_TestRandomAccessReaders, unittest.TestCase, VectorExampleMixin):
    def checkRead(self, reader):
        self.assertTrue(np.array_equal([3.0, 5.0, 7.0], reader["one"].numpy()))
        self.assertTrue(np.array_equal([1.0, 2.0, 3.0], reader["two"].numpy()))
        self.assertEqual(0, len(reader["three"].numpy()))

class TestRandomAccessMatrixReader(_TestRandomAccessReaders, unittest.TestCase, MatrixExampleMixin):
    def checkRead(self, reader):
        self.assertTrue(np.array_equal(np.arange(9).reshape((3, 3)), reader["one"].numpy()))
        self.assertTrue(np.array_equal([[1.0], [2.0], [3.0]], reader["two"].numpy()))
        self.assertEqual(0, len(reader["three"].numpy()))

class TestRandomAccessWaveReader(_TestRandomAccessReaders, unittest.TestCase, WaveExampleMixin):
    def checkRead(self, reader):
        self.assertTrue(np.array_equal(np.arange(9).reshape((3, 3)), reader["one"].data().numpy()))

class TestRandomAccessIntReader(_TestRandomAccessReaders, unittest.TestCase, IntExampleMixin):
    def checkRead(self, reader):
        self.assertEqual(1, reader['one'])
        self.assertEqual(3, reader['three'])
        self.assertEqual(2, reader['two'])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessFloatReader(_TestRandomAccessReaders, unittest.TestCase, IntExampleMixin):
    def checkRead(self, reader):
        self.assertEqual(1.0, reader['one'])
        self.assertEqual(3.0, reader['three'])
        self.assertEqual(2.0, reader['two'])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessBoolReader(_TestRandomAccessReaders, unittest.TestCase, BoolExampleMixin):
    def checkRead(self, reader):
        self.assertEqual(True, reader['one'])
        self.assertEqual(False, reader['three'])
        self.assertEqual(True, reader['two'])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessIntVectorReader(_TestRandomAccessReaders, unittest.TestCase, IntVectorExampleMixin):
    def checkRead(self, reader):
        self.assertEqual([1], reader['one'])
        self.assertEqual([], reader['three'])
        self.assertEqual([2, 3], reader['two'])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessIntVectorVectorReader(_TestRandomAccessReaders, unittest.TestCase, IntVectorVectorExampleMixin):
    def checkRead(self, reader):
        self.assertEqual([[1]], reader['one'])
        self.assertEqual([], reader['three'])
        self.assertEqual([[1, 2], [3, 4]], reader['two'])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessIntPairVectorReader(_TestRandomAccessReaders, unittest.TestCase, IntPairVectorExampleMixin):
    def checkRead(self, reader):
        self.assertEqual([(1, 1)], reader["one"])
        self.assertEqual([], reader["three"])
        self.assertEqual([(2, 3), (4, 5)], reader["two"])

        with self.assertRaises(KeyError):
            reader['four']

class TestRandomAccessFloatPairVectorReader(_TestRandomAccessReaders, unittest.TestCase, FloatPairVectorExampleMixin):
    def checkRead(self, reader):
        self.assertEqual([(1.0, 1.0)], reader['one'])
        self.assertEqual([], reader['three'])
        self.assertEqual([(2.0, 3.0), (4.0, 5.0)], reader['two'])

        with self.assertRaises(KeyError):
            reader['four']


if __name__ == '__main__':
    unittest.main()
