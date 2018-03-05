from __future__ import division
import unittest
import numpy as np

from kaldi.matrix import Matrix, SubMatrix, SubVector
from kaldi.matrix import DoubleMatrix, DoubleSubMatrix, DoubleSubVector
from kaldi.matrix.packed import SpMatrix, TpMatrix

class _Tests(object):

    def test_copy(self):
        m = self.matrix_class()
        m1 = self.matrix_class().copy_(m)
        self.assertTupleEqual((0, 0), m1.shape)

        m = self.matrix_class(5, 5)
        m1 = self.matrix_class(5, 5).copy_(m)
        self.assertTupleEqual((5, 5), m1.shape)

        m = self.matrix_class([[1., 2.], [3., 4.]])
        m1 = self.matrix_class(2, 2).copy_(m)
        self.assertEqual(m[0, 0], m1[0, 0])
        self.assertEqual(m[0, 1], m1[0, 1])
        self.assertEqual(m[1, 1], m1[1, 1])

        m[1, 1] = 5.0
        self.assertNotEqual(m[1, 1], m1[1, 1])

        with self.assertRaises(ValueError):
            m = self.matrix_class(5, 5)
            m1 = self.matrix_class(2, 2).copy_(m)

    def test_clone(self):
        m = self.matrix_class()
        m1 = m.clone()
        self.assertTupleEqual((0, 0), m1.shape)

        m = self.matrix_class(5, 5)
        m1 = m.clone()
        self.assertTupleEqual((5, 5), m1.shape)

        m = self.matrix_class([[1., 2.], [3., 4.]])
        m1 = m.clone()

        self.assertEqual(m[0, 0], m1[0, 0])
        self.assertEqual(m[0, 1], m1[0, 1])
        self.assertEqual(m[1, 1], m1[1, 1])

        m[1, 1] = 5.0
        self.assertNotEqual(m[1, 1], m1[1, 1])

    def test_size(self):
        m = self.matrix_class()
        self.assertTupleEqual((0, 0), m.size())

        m = self.matrix_class(10, 10)
        self.assertTupleEqual((10, 10), m.size())

    def test_equal(self):
        m = self.matrix_class()
        self.assertTrue(m == m.clone())

        m = self.matrix_class(4, 4)
        m.set_zero_()
        m1 = self.matrix_class(4, 4)
        m1.set_zero_()
        self.assertTrue(m == m1)

        m = self.matrix_class(4, 4)
        m1 = SpMatrix(4)
        self.assertFalse(m == m1)

    def test_numpy(self):
        m = self.matrix_class()
        n = m.numpy()
        self.assertTupleEqual((0, 0), n.shape)

        m = self.matrix_class(5, 5)
        n = m.numpy()
        self.assertTupleEqual((5, 5), n.shape)

        m = self.matrix_class([[1.0, -2.0], [3.0, -4.0]])
        n = m.numpy()
        self.assertTupleEqual((2, 2), n.shape)
        self.assertEqual(1.0, n[0, 0])
        self.assertEqual(-2.0, n[0, 1])
        self.assertEqual(-4.0, n[1, 1])

        # Test __array__
        n = np.asarray(m)
        self.assertIsInstance(n, np.ndarray)
        # self.assertEqual(n.dtype, np.float32)
        for i in range(m.num_rows):
            for j in range(m.num_cols):
                self.assertEqual(m[i,j], n[i,j])

        # Test __array__wrap__
        abs_m = np.abs(m)
        abs_n = np.abs(n)
        self.assertIsInstance(abs_m, (SubMatrix, DoubleSubMatrix))
        for i in range(m.num_rows):
            for j in range(m.num_cols):
                self.assertEqual(abs_m[i,j], abs_n[i,j])

        # Test some ufuncs
        for func in ['sin', 'exp', 'square']:
            ufunc = getattr(np, func)
            res_m = ufunc(m)
            res_n = ufunc(n)
            self.assertIsInstance(res_m, (SubMatrix, DoubleSubMatrix))
            for i in range(m.num_rows):
                for j in range(m.num_cols):
                    self.assertEqual(res_m[i,j], res_n[i,j])

        # Test a ufunc with boolean return value
        geq0_m = np.greater_equal(m, 0)
        geq0_n = np.greater_equal(n, 0).astype('float32')
        self.assertIsInstance(geq0_m, (SubMatrix, DoubleSubMatrix))
        for i in range(m.num_rows):
            for j in range(m.num_cols):
                self.assertEqual(geq0_m[i,j], geq0_n[i,j])

        # Test a ufunc method that change the number of dimensions
        max0_m = np.maximum.reduce(m)
        max0_n = np.maximum.reduce(n)
        self.assertIsInstance(max0_m, (SubVector, DoubleSubVector))
        for i in range(len(max0_m)):
            self.assertEqual(max0_m[i], max0_n[i])

    def test_range(self):
        m = self.matrix_class()

        self.assertTupleEqual((0, 0), m.range(0, 0, 0, 0).size())

        with self.assertRaises(IndexError):
            m.range(0, 1, 0, 0)

        with self.assertRaises(IndexError):
            m.range(0, 0, 0, 1)

        m = self.matrix_class([[1.0, 2.0], [3.0, 4.0]])
        s = m.range(0, None, 0, None)
        self.assertTupleEqual((2, 2), s.shape)

    def test__getitem__(self):
        m = self.matrix_class([[3, 5], [7, 11]])
        self.assertAlmostEqual(3.0, m[0, 0])
        self.assertAlmostEqual(5.0, m[0, 1])
        self.assertAlmostEqual(7.0, m[1, 0])
        self.assertAlmostEqual(11.0, m[1, 1])

        with self.assertRaises(IndexError):
            m[3, 0]

        with self.assertRaises(IndexError):
            m[0, 3]

        self.assertAlmostEqual(15.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(77.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(21.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(55.0, m[:, 1].numpy().prod())

    def test__setitem__(self):
        m = self.matrix_class()
        with self.assertRaises(IndexError):
            m[0] = 1.0

        m = self.matrix_class(2, 2)
        m[0, 0] = 3.0
        m[0, 1] = 5.0
        m[1, 0] = 7.0
        m[1, 1] = 11.0
        self.assertAlmostEqual(3.0, m[0, 0])
        self.assertAlmostEqual(5.0, m[0, 1])
        self.assertAlmostEqual(7.0, m[1, 0])
        self.assertAlmostEqual(11.0, m[1, 1])

        with self.assertRaises(IndexError):
            m[2, 0] = 10.0

        with self.assertRaises(IndexError):
            m[0, 2] = 10.0

        m = self.matrix_class([[3, 5], [7, 11]])
        m[0, 0] = 13.0

        self.assertAlmostEqual(65.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(77.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(91.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(55.0, m[:, 1].numpy().prod())


        m = self.matrix_class([[3, 5], [7, 11]])
        m[0, :] = 3.0

        self.assertAlmostEqual(9.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(77.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(21.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(33.0, m[:, 1].numpy().prod())

        m = self.matrix_class([[3, 5], [7, 11]])
        m[:, 0] = 3.0

        self.assertAlmostEqual(15.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(33.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(9.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(55.0, m[:, 1].numpy().prod())

    def test__init__(self):
        # Test Empty
        m = self.matrix_class()
        self.assertIsNotNone(m)
        self.assertTupleEqual((0, 0), m.size())

        # Missing dim
        with self.assertRaises(IndexError):
            m = self.matrix_class([[]])

        # num_cols == 0 but num_rows != 0
        with self.assertRaises(IndexError):
            m = self.matrix_class([[], []])

        # Construct with size
        m = self.matrix_class(100, 100)
        self.assertIsNotNone(m)
        self.assertTupleEqual((100, 100), m.size())

        # Construct with list
        m = self.matrix_class([[3, 5], [7, 11]])
        self.assertIsNotNone(m)
        self.assertTupleEqual((2, 2), m.size())

        # Construct with np.array
        m2 = self.matrix_class(np.array([[3, 5], [7, 11]]))
        self.assertIsNotNone(m2)
        self.assertTupleEqual((2, 2), m2.size())

        self.assertTrue(m.equal(m2))
        self.assertTrue(m2.equal(m))

        # Construct with other np.array
        m = self.matrix_class(np.arange(10).reshape(-1, 2))
        self.assertTupleEqual((5, 2), m.size())

        with self.assertRaises(ValueError):
            m = self.matrix_class(np.empty((1, 1, 1)))

        # Construct with other matrix
        m2 = self.matrix_class(m)
        self.assertTupleEqual((5, 2), m2.size())

        # Construct with TpMatrix
        m2 = self.matrix_class(TpMatrix(10))
        self.assertTupleEqual((10, 10), m2.size())

    def test__delitem__(self):
        m = self.matrix_class()
        with self.assertRaises(IndexError):
            del m[0]

        m = self.matrix_class([[3, 5], [7, 11]])

        del m[0] #deletes row 0

        self.assertTupleEqual((1, 2), m.size())
        self.assertAlmostEqual(7, m[0,0])
        self.assertAlmostEqual(11, m[0,1])

class testSubMatrix(unittest.TestCase):

    def test__init__(self):
        m = Matrix()
        sb = SubMatrix(m)

        m = Matrix(5, 5)
        sb = SubMatrix(m)

        for i in range(100):
            m.set_randn_()
            self.assertAlmostEqual(m.sum(), sb.sum())

        m = DoubleMatrix()
        sb = SubMatrix(m)

class testSingleMatrix(unittest.TestCase, _Tests):
    matrix_class = Matrix

class testDoubleMatrix(unittest.TestCase, _Tests):
    matrix_class = DoubleMatrix

if __name__ == '__main__':
    unittest.main()
