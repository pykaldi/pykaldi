from __future__ import division
import unittest
import numpy as np
from kaldi.matrix import Vector, SubVector

class TestVector(unittest.TestCase):

    @unittest.skip("TODO")
    def testErrors(self):
        pass

    def test_copy(self):
        v = Vector(5)
        with self.assertRaises(ValueError):
            v1 = Vector().copy(v)

        v.set_zero()
        v1 = Vector(len(v)).copy(v)
        self.assertEqual(len(v), len(v1))

        # Make sure modifying original
        # doesn't break new one
        v[0] = 1.0
        v[1] = 2.0
        v[2] = 3.0

        self.assertNotEqual(v[0], v1[0])
        self.assertNotEqual(v[1], v1[1])
        self.assertNotEqual(v[2], v1[2])

        # Check copy works with data
        v1 = Vector(len(v)).copy(v)

        self.assertEqual(v[0], v1[0])
        self.assertEqual(v[1], v1[1])
        self.assertEqual(v[2], v1[2])

    def test_clone(self):

        # Empty clone
        v = Vector()
        v2 = v.clone()

        # Clone with data
        v = Vector.new(np.array([3, 5, 7]))
        v2 = v.clone()

        self.assertEqual(v[0], v2[0])
        self.assertEqual(v[1], v2[1])
        self.assertEqual(v[2], v2[2])

        # Make sure modifying original
        # doesn't break new one
        v.set_zero()
        self.assertNotEqual(v[0], v2[0])
        self.assertNotEqual(v[1], v2[1])
        self.assertNotEqual(v[2], v2[2])

    def test_shape(self):
        v = Vector()
        self.assertTupleEqual((0,), v.shape)

        v = Vector(5)
        self.assertTupleEqual((5,), v.shape)

    def test_equal(self):
        v = Vector()
        v1 = Vector()
        self.assertEqual(v, v1)

        v = Vector(5)
        v1 = Vector(4)
        self.assertNotEqual(v, v1)

        v = Vector(5)
        v1 = Vector(5)

        v[0] = 10.0
        v1[0] = 11.0
        self.assertNotEqual(v, v1)

        v[0] = v1[0]
        self.assertTrue(v == v1)

    def test_numpy(self):
        v = Vector()
        v1 = v.numpy()
        self.assertTupleEqual((0, ), v1.shape)

        v = Vector(5)
        v1 = v.numpy()
        self.assertTupleEqual((5, ), v1.shape)

        v = Vector.new([1.0, -2.0, 3.0])
        v1 = v.numpy()
        self.assertTrue(np.all(np.array([1.0, -2.0, 3.0]) == v1))

        # Test __array__
        n = np.asarray(v)
        self.assertIsInstance(n, np.ndarray)
        self.assertEqual(n.dtype, np.float32)
        for i in range(len(v)):
            self.assertEqual(v[i], n[i])

        # Test __array__wrap__
        abs_v = np.abs(v)
        abs_n = np.abs(n)
        self.assertIsInstance(abs_v, SubVector)
        for i in range(len(v)):
            self.assertEqual(abs_v[i], abs_n[i])

        # Test some ufuncs
        for func in ['sin', 'exp', 'square']:
            ufunc = getattr(np, func)
            res_v = ufunc(v)
            res_n = ufunc(n)
            self.assertIsInstance(res_v, SubVector)
            for i in range(len(v)):
                self.assertEqual(res_v[i], res_n[i])

        # Test a ufunc with boolean return value
        geq0_v = np.greater_equal(v, 0)
        geq0_n = np.greater_equal(n, 0).astype('float32')
        self.assertIsInstance(geq0_v, SubVector)
        for i in range(len(v)):
            self.assertEqual(geq0_v[i], geq0_n[i])

    def test_range(self):
        v = Vector.new(np.array([3, 5, 7, 11, 13]))
        v1 = v.range(1, 2)

        self.assertTrue(isinstance(v1, SubVector))
        self.assertTrue(2, v1.size())
        v2 = Vector.new([5, 7])
        self.assertEqual(v2, v1)

        # What happens if we modify v?
        v[0] = -1.0
        self.assertEqual(v2, v1)
        self.assertEqual(v.range(1, 2), v2)

    def test_empty(self):
        # Test empty kaldi.Vector
        v = Vector()
        self.assertIsNotNone(v)
        self.assertEqual(0, v.size())

        v = Vector.new([])

    def test_nonempty(self):
        v = Vector(100)
        self.assertIsNotNone(v)
        self.assertEqual(100, v.size())

    def test_new(self):
        v = Vector.new([3, 5, 7, 11, 13])
        self.assertEqual(5, v.size())
        self.assertAlmostEqual(15015.0, v.numpy().prod())

        v2 = Vector.new(np.array([3, 5, 7, 11, 13]))
        self.assertEqual(5, v2.size())
        self.assertAlmostEqual(15015.0, v2.numpy().prod())

        self.assertTrue(v.equal(v2))
        self.assertTrue(v2.equal(v))

    def test__getitem__(self):
        v = Vector()
        with self.assertRaises(IndexError):
            v[0]

        v = Vector.new([3, 5, 7, 11, 13])
        self.assertAlmostEqual(3.0, v[0])
        self.assertAlmostEqual(7.0, v[2])
        self.assertAlmostEqual(13.0, v[4])

        with self.assertRaises(IndexError):
            v[5]

        self.assertAlmostEqual(15015.0, v[:10].numpy().prod())
        self.assertAlmostEqual(15015.0, v[::-1].numpy().prod())
        self.assertAlmostEqual(1001.0, v[2:5].numpy().prod())

    def test__setitem__(self):
        v = Vector()
        with self.assertRaises(IndexError):
            v[0] = 1.0

        v = Vector.new([3, 5, 7, 11, 13])
        v[0] = 15.0
        self.assertAlmostEqual(75075.0, v[:10].numpy().prod())

        with self.assertRaises(ValueError):
            v[0:3] = np.array([3, 5, 7, 11, 13])

        v[0:5] = np.array([3, 5, 7, 11, 13])
        self.assertAlmostEqual(15015.0, v.numpy().prod())

    def test__delitem__(self):
        v = Vector()
        with self.assertRaises(IndexError):
            del v[0]

        v = Vector.new([3, 5, 7, 11, 13])
        del v[0]
        self.assertAlmostEqual(5005.0, v.numpy().prod())

if __name__ == '__main__':
    unittest.main()
