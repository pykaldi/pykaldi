from kaldi.cudamatrix import *
from kaldi.base import math as kaldi_math
from kaldi.matrix import Vector, Matrix

import unittest
import numpy as np 

class TestCuMatrix(unittest.TestCase):

    def testNew(self):
        A = CuMatrix()
        self.assertIsNotNone(A)
        self.assertEqual(0, A.NumRows())
        self.assertEqual(0, A.NumCols())

        dim = A.Dim()
        self.assertEqual(0, dim.rows)
        self.assertEqual(0, dim.cols)

        A = CuMatrix.new_from_size(10, 10)
        self.assertIsNotNone(A)
        self.assertEqual(10, A.NumRows())
        self.assertEqual(10, A.NumCols())

        dim = A.Dim()
        self.assertEqual(10, dim.rows)
        self.assertEqual(10, dim.cols)

        A = CuMatrix.new_from_matrix(Matrix.new([[2, 3], [5, 7]]))
        self.assertIsNotNone(A)
        self.assertEqual(2, A.NumRows())
        self.assertEqual(2, A.NumCols())

        B = CuMatrix.new_from_other(A)
        self.assertIsNotNone(B)
        self.assertEqual(2, B.NumRows())
        self.assertEqual(2, B.NumCols())

    def testResize(self):
        A = CuMatrix()
        A.Resize(10, 10)
        self.assertEqual(10, A.NumRows())
        self.assertEqual(10, A.NumCols())

        # A.Resize(-1, -1) #This hard-crashes
        A.Resize(0, 0)

        A = CuMatrix.new_from_matrix(Matrix.new([[1, 2], [3, 4], [5, 6]])) #A is 3x2
        with self.assertRaises(Exception):
            A.Resize(2, 2) #Try to resize to something invalid 

    # FIXME:
    # Hard crashing...
    @unittest.skip("hard-crashes")
    def testSwap(self):
        for i in range(10):
            dim = (10 * i, 4 * i)
            M = Matrix.new(np.random.random(dim)).clone()
            A = CuMatrix.new_from_matrix(M)
            B = CuMatrix.new_from_size(A.NumRows(), A.NumCols())
            B.Swap(A)
            self.assertAlmostEqual(A.Sum(), B.Sum(), places = 4) #Kaldi's precision is aweful
            self.assertAlmostEqual(M.sum(), B.Sum(), places = 4) #Kaldi's precision is aweful

            C = CuMatrix.new_from_size(M.shape[0], M.shape[1])
            C.SwapWithMatrix(M)
            self.assertAlmostEqual(B.Sum(), C.Sum(), places = 4) #Kaldi's precision is aweful

    def testCopyFromMat(self):
        for i in range(10):
            rows, cols = 10*i, 5*i
            A = Matrix(rows, cols)
            A.SetRandn()
            B = CuMatrix.new_from_size(*A.shape)
            B.CopyFromMat(A)
            self.assertAlmostEqual(A.Sum(), B.Sum(), places = 4)
    
            A = CuMatrix.new_from_size(rows, cols)
            A.SetRandn()
            B = CuMatrix.new_from_size(rows, cols)
            B.CopyFromCuMat(A)
            self.assertAlmostEqual(A.Sum(), B.Sum(), places = 4)

    @unittest.skip("hard-crashes")
    def test__getitem(self):
        A = CuMatrix.new_from_matrix(Matrix.new(np.arange(10).reshape((5, 2))))
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
        self.assertTrue(SameDimCuMatrix(A, B))

        A = CuMatrix.new_from_size(10, 10)
        B = CuMatrix.new_from_size(10, 9)
        self.assertFalse(SameDimCuMatrix(A, B))

    def testApproxEqual(self):
        A = CuMatrix()
        B = CuMatrix()
        self.assertTrue(ApproxEqualCuMatrix(A, B))

        A.SetZero()
        B.SetZero()
        self.assertTrue(ApproxEqualCuMatrix(A, B))

        B.SetRandn()
        B.Scale(10.0)
        self.assertFalse(ApproxEqualCuMatrix(A, B))

if __name__ == '__main__':
    for i in range(2):
        CuDevice.Instantiate().SetDebugStrideMode(True)
        if i == 0:
            CuDevice.Instantiate().SelectGpuId("no")
        else:
            CuDevice.Instantiate().SelectGpuId("yes")
        
        unittest.main()

        CuDevice.Instantiate().PrintProfile()