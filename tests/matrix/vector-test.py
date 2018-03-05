from __future__ import division
import unittest
import numpy as np
from kaldi.matrix import Vector, SubVector
from kaldi.matrix import DoubleVector, DoubleSubVector
from kaldi.matrix import Matrix, DoubleMatrix

class _Tests(object):

    def test_copy(self):
        v = self.vector_class(5)
        with self.assertRaises(ValueError):
            v1 = self.vector_class().copy_(v)

        v1 = self.vector_class(len(v)).copy_(v)
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
        v1 = self.vector_class(len(v)).copy_(v)

        self.assertEqual(v[0], v1[0])
        self.assertEqual(v[1], v1[1])
        self.assertEqual(v[2], v1[2])

        for i in range(10):
            v = self.vector_class(i).set_randn_()
            v1 = self.vector_class(i).copy_(v)
            self.assertEqual(v, v1)

        v = DoubleVector(5).set_randn_()
        v1 = self.vector_class(5).copy_(v)
        # self.assertEqual(v, v1) #This fails due to precision/type

    def test_clone(self):

        # Empty clone
        v = self.vector_class()
        v2 = v.clone()

        # Clone with data
        v = self.vector_class(np.array([3, 5, 7]))
        v2 = v.clone()

        self.assertEqual(v[0], v2[0])
        self.assertEqual(v[1], v2[1])
        self.assertEqual(v[2], v2[2])

        # Make sure modifying original
        # doesn't break new one
        v.set_zero_()
        self.assertNotEqual(v[0], v2[0])
        self.assertNotEqual(v[1], v2[1])
        self.assertNotEqual(v[2], v2[2])

    def test_shape(self):
        v = self.vector_class()
        self.assertTupleEqual((0,), v.shape)

        v = self.vector_class(5)
        self.assertTupleEqual((5,), v.shape)

    def test_equal(self):
        v = self.vector_class()
        v1 = self.vector_class()
        self.assertEqual(v, v1)

        v = self.vector_class(5)
        v1 = self.vector_class(4)
        self.assertNotEqual(v, v1)

        v = self.vector_class(5)
        v1 = self.vector_class(5)

        v[0] = 10.0
        v1[0] = 11.0
        self.assertNotEqual(v, v1)

        v[0] = v1[0]
        self.assertTrue(v == v1)

    def test_numpy(self):
        v = self.vector_class()
        v1 = v.numpy()
        self.assertTupleEqual((0, ), v1.shape)

        v = self.vector_class(5)
        v1 = v.numpy()
        self.assertTupleEqual((5, ), v1.shape)

        v = self.vector_class([1.0, -2.0, 3.0])
        v1 = v.numpy()
        self.assertTrue(np.all(np.array([1.0, -2.0, 3.0]) == v1))

        # Test __array__
        n = np.asarray(v)
        self.assertIsInstance(n, np.ndarray)
        # self.assertEqual(n.dtype, np.float32)
        for i in range(len(v)):
            self.assertEqual(v[i], n[i])

        # Test __array__wrap__
        abs_v = np.abs(v)
        abs_n = np.abs(n)
        self.assertIsInstance(abs_v, (SubVector, DoubleSubVector))
        for i in range(len(v)):
            self.assertEqual(abs_v[i], abs_n[i])

        # Test some ufuncs
        for func in ['sin', 'exp', 'square']:
            ufunc = getattr(np, func)
            res_v = ufunc(v)
            res_n = ufunc(n)
            self.assertIsInstance(res_v, (SubVector, DoubleSubVector))
            for i in range(len(v)):
                self.assertEqual(res_v[i], res_n[i])

        # Test a ufunc with boolean return value
        geq0_v = np.greater_equal(v, 0)
        geq0_n = np.greater_equal(n, 0).astype('float32')
        self.assertIsInstance(geq0_v, (SubVector, DoubleSubVector))
        for i in range(len(v)):
            self.assertEqual(geq0_v[i], geq0_n[i])

    def test_range(self):
        v = self.vector_class(np.array([3, 5, 7, 11, 13]))
        v1 = v.range(1, 2)

        self.assertTrue(isinstance(v1, (SubVector, DoubleSubVector)))
        self.assertTrue(2, len(v1))
        v2 = self.vector_class([5, 7])
        self.assertEqual(v2, v1)

        # What happens if we modify v?
        v[0] = -1.0
        self.assertEqual(v2, v1)
        self.assertEqual(v.range(1, 2), v2)

    def add_vec_(self):
        v = self.vector_class(5).set_randn_()
        v1 = self.vector_class(5).set_zero_()
        self.assertEqual(v, v.add_vec_(v1))

        v1 = v1.set_randn_()
        self.assertNotEqual(v, v.add_vec_(v1))

        v1 = v.clone()
        v1 = v1.scale_(-1.0)
        self.assertEqual(Vector(5), v.add_vec_(1.0, v1))

    def test_copy_row_from_mat(self):
        
        with self.assertRaises(IndexError):
            M = Matrix(0, 0).set_randn_()
            v = self.vector_class(0).copy_row_from_mat_(M, 0)

        for i in range(1, 11):
            M = Matrix(i, i).set_randn_()
            v = self.vector_class(i).copy_row_from_mat_(M, 0)
            for m, e in zip(M[0], v):
                self.assertEqual(m, e)

    def test_init(self):

        # Construct empty
        v = self.vector_class()
        self.assertIsNotNone(v)
        self.assertEqual(0, len(v))

        v = self.vector_class([])

        # Construct with size
        v = self.vector_class(100)
        self.assertIsNotNone(v)
        self.assertEqual(100, len(v))

        with self.assertRaises(ValueError):
            v = self.vector_class(-100)

        # Construct with list
        v = self.vector_class([3, 5, 7, 11, 13])
        self.assertEqual(5, len(v))
        self.assertAlmostEqual(15015.0, v.numpy().prod())

        # Construct with np.array
        v2 = self.vector_class(np.array([3, 5, 7, 11, 13]))
        self.assertEqual(5, len(v2))
        self.assertAlmostEqual(15015.0, v2.numpy().prod())
        self.assertEqual(v, v2)

        # Construct with other np.array
        v = self.vector_class(np.arange(10))
        self.assertEqual(10, v.dim)

        with self.assertRaises(TypeError):
            v = self.vector_class(np.arange(10).reshape(-1, 2))

        # New with double
        v = self.vector_class(DoubleVector(10).set_randn_())
        self.assertEqual(10, v.dim)

    def test__getitem__(self):
        v = self.vector_class()
        with self.assertRaises(IndexError):
            v[0]

        v = self.vector_class([3, 5, 7, 11, 13])
        self.assertAlmostEqual(3.0, v[0])
        self.assertAlmostEqual(7.0, v[2])
        self.assertAlmostEqual(13.0, v[4])

        with self.assertRaises(IndexError):
            v[5]

        self.assertAlmostEqual(15015.0, v[:10].numpy().prod())
        self.assertAlmostEqual(15015.0, v[::-1].numpy().prod())
        self.assertAlmostEqual(1001.0, v[2:5].numpy().prod())

    def test__setitem__(self):
        v = self.vector_class()
        with self.assertRaises(IndexError):
            v[0] = 1.0

        v = self.vector_class([3, 5, 7, 11, 13])
        v[0] = 15.0
        self.assertAlmostEqual(75075.0, v[:10].numpy().prod())

        with self.assertRaises(ValueError):
            v[0:3] = np.array([3, 5, 7, 11, 13])

        v[0:5] = np.array([3, 5, 7, 11, 13])
        self.assertAlmostEqual(15015.0, v.numpy().prod())

    def test__delitem__(self):
        v = self.vector_class()
        with self.assertRaises(IndexError):
            del v[0]

        v = self.vector_class([3, 5, 7, 11, 13])
        del v[0]
        self.assertAlmostEqual(5005.0, v.numpy().prod())


class SingleVectorTest(unittest.TestCase, _Tests):
    vector_class = Vector

class DoubleVectorTest(unittest.TestCase, _Tests):
    vector_class = DoubleVector

if __name__ == '__main__':
    unittest.main()