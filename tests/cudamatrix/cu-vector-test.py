from kaldi.base import math as kaldi_math
from kaldi.matrix import Vector, TpMatrix
from kaldi.cudamatrix import *

import unittest
import numpy as np 

class TestCuVector(unittest.TestCase):
    def testCuVectorNewFromSize(self):
        vec = CuVector()
        self.assertIsNotNone(vec)
        self.assertEqual(0, vec.Dim())

        for i in range(10):
            dim = 10 * i 
            vec = CuVector.new_from_size(dim)
            self.assertIsNotNone(vec)
            self.assertEqual(dim, vec.Dim())

    def testCuVectorResize(self):
        for i in range(10):
            dim = 10 * i 
            vec = CuVector()
            vec.Resize(dim)
            self.assertEqual(dim, vec.Dim())

    @unittest.skip("TODO")
    def testCuVectorRead(self):
        pass 

    @unittest.skip("TODO")
    def testCuVectorWrite(self):
        pass 

    def testCuVectorSwap(self):
        N = [2, 3, 5, 7, 13]
        A = Vector.new(N).clone()
        C = CuVector.new_from_size(5)
        C.Swap(A) #Swap *is* destructive

        self.assertEqual(16.0, C.Norm(2))

        A = Vector()
        C = CuVector.new_from_size(0)
        C.Swap(A)
        self.assertEqual(0.0, C.Norm(2))

    def testCuVectorCopyFromVec(self):

        # Shouldnt crash
        A = Vector()
        C = CuVector.new_from_size(0)
        C.CopyFromVec(A)

        # What if dims not match?
        # HARD-CRASH
        # FIXME
        # A = Vector.random(10)
        # C = CuVector.new_from_size(0)
        # C.CopyFromVec(A)

        for i in range(10):
            dim = 10 * i 
            A = Vector(dim)
            A.SetRandn()
            D = CuVector.new_from_size(dim)
            D.CopyFromVec(A)
            self.assertEqual(A.Sum(), D.Sum())

    def testCuSubVector(self):
        for iteration in range(10):
            M1 = 1 + kaldi_math.Rand() % 10
            M2 = 1 + kaldi_math.Rand() % 1
            M3 = 1 + kaldi_math.Rand() % 10
            M = M1 + M2 + M3 

            m = kaldi_math.Rand() % M2

            vec = CuVector.new_from_size(M)
            vec.SetRandn()

            subvec1 = CuSubVector(vec, M1, M2)
            # subvec2 = vec.range(M1, M2)

            f1, f2, f3 = vec[M1 + m], subvec1[m], subvec2[m]
            self.assertEqual(f1, f2)
            self.assertEqual(f3, f2)

    def testCuVectorInverElements(self):
        # Test that this doesnt crash
        C = CuVector.new_from_size(0)
        C.InvertElements()

        C = CuVector.new_from_size(10)
        C.SetRandn()
        C.InvertElements()

        # Geometric series r = 1/2, a = 1/2
        A = Vector.new([2, 4, 8, 16, 32, 64])
        C = CuVector.new_from_size(len(A))
        C.Swap(A)
        C.InvertElements()

        f1 = C.Sum()
        self.assertAlmostEqual(0.5 * (1 - 0.5**len(A))/(1 - 0.5), f1)

    def testCuVectorGetItem(self):
        v = CuVector()
        with self.assertRaises(IndexError):
            v[0]

        v = CuVector.new([3, 5, 7, 11, 13])
        self.assertAlmostEqual(3.0, v[0])
        self.assertAlmostEqual(7.0, v[2])
        self.assertAlmostEqual(13.0, v[4])

        with self.assertRaises(IndexError):
            v[5]

if __name__ == '__main__':
    for i in range(2):
        CuDevice.Instantiate().SetDebugStrideMode(True)
        if i == 0:
            CuDevice.Instantiate().SelectGpuId("no")
        else:
            CuDevice.Instantiate().SelectGpuId("yes")
        
        unittest.main()

        CuDevice.Instantiate().PrintProfile()