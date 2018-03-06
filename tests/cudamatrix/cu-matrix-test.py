from kaldi.base import math as kaldi_math
from kaldi.matrix import Vector, Matrix

from kaldi.cudamatrix import (CuMatrix, CuVector,
                              approx_equal_cu_matrix, same_dim_cu_matrix)

import unittest
import numpy as np

class TestCuMatrix(unittest.TestCase):

    def testNew(self):
        A = CuMatrix()
        self.assertIsNotNone(A)
        self.assertEqual(0, A.num_rows())
        self.assertEqual(0, A.num_cols())

        dim = A.dim()
        self.assertEqual(0, dim.rows)
        self.assertEqual(0, dim.cols)

        A = CuMatrix.from_size(10, 10)
        self.assertIsNotNone(A)
        self.assertEqual(10, A.num_rows())
        self.assertEqual(10, A.num_cols())

        dim = A.dim()
        self.assertEqual(10, dim.rows)
        self.assertEqual(10, dim.cols)

        A = CuMatrix.from_matrix(Matrix([[2, 3], [5, 7]]))
        self.assertIsNotNone(A)
        self.assertEqual(2, A.num_rows())
        self.assertEqual(2, A.num_cols())

        B = CuMatrix.from_other(A)
        self.assertIsNotNone(B)
        self.assertEqual(2, B.num_rows())
        self.assertEqual(2, B.num_cols())

    def testResize(self):
        A = CuMatrix()
        A.resize(10, 10)
        self.assertEqual(10, A.num_rows())
        self.assertEqual(10, A.num_cols())

        # A.resize(-1, -1) #This hard-crashes
        A.resize(0, 0)

        # TODO:
        # A = CuMatrix.from_matrix(Matrix.new([[1, 2], [3, 4], [5, 6]])) #A is 3x2
        # with self.assertRaises(Exception):
        #     A.resize(2, 2) #Try to resize to something invalid

    # FIXME:
    # Hard crashing...
    @unittest.skip("hard-crashes")
    def testSwap(self):
        for i in range(10):
            dim = (10 * i, 4 * i)
            M = Matrix(np.random.random(dim))
            A = CuMatrix.from_matrix(M)
            B = CuMatrix.from_size(A.num_rows(), A.num_cols())
            B.Swap(A)
            self.assertAlmostEqual(A.sum(), B.sum(), places = 4) #Kaldi's precision is aweful
            self.assertAlmostEqual(M.sum(), B.sum(), places = 4) #Kaldi's precision is aweful

            C = CuMatrix.from_size(M.shape[0], M.shape[1])
            C.SwapWithMatrix(M)
            self.assertAlmostEqual(B.sum(), C.sum(), places = 4) #Kaldi's precision is aweful

    def testcopy_from_mat(self):
        for i in range(1, 10):
            rows, cols = 10*i, 5*i
            A = Matrix(rows, cols)
            A.set_randn_()
            B = CuMatrix.from_size(*A.shape)
            B.copy_from_mat(A)
            self.assertAlmostEqual(A.sum(), B.sum(), places = 4)

            A = CuMatrix.from_size(rows, cols)
            A.set_randn()
            B = CuMatrix.from_size(rows, cols)
            B.copy_from_cu_mat(A)
            self.assertAlmostEqual(A.sum(), B.sum(), places = 4)

    @unittest.skip("hard-crashes")
    def test__getitem(self):
        A = CuMatrix.from_matrix(Matrix.new(np.arange(10).reshape((5, 2))))
        self.assertEqual(0.0, A.__getitem(0, 0))
        self.assertEqual(1.0, A.__getitem(0, 1))
        self.assertEqual(2.0, A.__getitem(1, 0))
        self.assertEqual(3.0, A.__getitem(1, 1))
        self.assertEqual(4.0, A.__getitem(2, 0))

        # This should hard crash
        with self.assertRaises(IndexError):
            self.assertEqual(0.0, A.__getitem(0, 2))

    def testSameDim(self):
        A = CuMatrix()
        B = CuMatrix()
        self.assertTrue(same_dim_cu_matrix(A, B))

        A = CuMatrix.from_size(10, 10)
        B = CuMatrix.from_size(10, 9)
        self.assertFalse(same_dim_cu_matrix(A, B))

    @unittest.skip("FIXME")
    def testApproxEqual(self):
        A = CuMatrix()
        B = CuMatrix()
        self.assertTrue(approx_equal_cu_matrix(A, B))

        A.SetZero()
        B.SetZero()
        self.assertTrue(approx_equal_cu_matrix(A, B))

        B.set_randn()
        B.Scale(10.0)
        self.assertFalse(approx_equal_cu_matrix(A, B))

if __name__ == '__main__':
    unittest.main()
