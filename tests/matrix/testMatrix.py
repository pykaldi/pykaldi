from __future__ import division
import unittest
import numpy as np 
from kaldi.matrix import Matrix, SubMatrix, SpMatrix

class TestKaldiMatrix(unittest.TestCase):

    def test_copy_(self):
        m = Matrix()
        m1 = Matrix().copy_(m)
        self.assertTupleEqual((0, 0), m1.shape)

        m = Matrix(5, 5)
        m1 = Matrix(5, 5).copy_(m)
        self.assertTupleEqual((5, 5), m1.shape)

        m = Matrix.new([[1., 2.], [3., 4.]])
        m1 = Matrix(2, 2).copy_(m)
        self.assertEqual(m[0, 0], m1[0, 0])
        self.assertEqual(m[0, 1], m1[0, 1])
        self.assertEqual(m[1, 1], m1[1, 1])

        m[1, 1] = 5.0
        self.assertNotEqual(m[1, 1], m1[1, 1])

        with self.assertRaises(ValueError):
            m = Matrix(5, 5)
            m1 = Matrix(2, 2).copy_(m)

    def test_clone(self):
        m = Matrix()
        m1 = m.clone()
        self.assertTupleEqual((0, 0), m1.shape)

        m = Matrix(5, 5)
        m1 = m.clone()
        self.assertTupleEqual((5, 5), m1.shape)
        
        m = Matrix.new([[1., 2.], [3., 4.]])
        m1 = m.clone()

        self.assertEqual(m[0, 0], m1[0, 0])
        self.assertEqual(m[0, 1], m1[0, 1])
        self.assertEqual(m[1, 1], m1[1, 1])

        m[1, 1] = 5.0
        self.assertNotEqual(m[1, 1], m1[1, 1])

    def test_size(self):
        m = Matrix()
        self.assertTupleEqual((0, 0), m.size())

        m = Matrix(10, 10)
        self.assertTupleEqual((10, 10), m.size())

    def test_equal(self):
        m = Matrix()
        self.assertTrue(m, m.clone())

        m = Matrix(4, 4)
        m.SetZero()
        m1 = Matrix(4, 4)
        m1.SetZero()
        self.assertTrue(m, m1)

        m = Matrix(4, 4)
        m1 = SpMatrix(4)
        self.assertFalse(m == m1)

    def test_numpy(self):
        m = Matrix()
        n = m.numpy()
        self.assertTupleEqual((0, 0), n.shape)

        m = Matrix(5, 5)
        n = m.numpy()
        self.assertTupleEqual((5, 5), n.shape)

        m = Matrix.new([[1.0, 2.0], [3.0, 4.0]])
        n = m.numpy()
        self.assertTupleEqual((2, 2), n.shape)
        self.assertEqual(1.0, n[0, 0])
        self.assertEqual(2.0, n[0, 1])
        self.assertEqual(4.0, n[1, 1])

    def test_range(self):
        m = Matrix()

        self.assertTupleEqual((0, 0), m.range(0, 0, 0, 0).size())

        with self.assertRaises(IndexError):
            m.range(0, 1, 0, 0)

        with self.assertRaises(IndexError):
            m.range(0, 0, 0, 1)

        m = Matrix.new([[1.0, 2.0], [3.0, 4.0]])
        s = m.range(0, None, 0, None)
        self.assertTupleEqual((2, 2), s.shape)

    def test__getitem__(self):
        m = Matrix.new([[3, 5], [7, 11]])
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
        m = Matrix()
        with self.assertRaises(IndexError):
            m[0] = 1.0

        m = Matrix(2, 2)
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

        m = Matrix.new([[3, 5], [7, 11]])
        m[0, 0] = 13.0

        self.assertAlmostEqual(65.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(77.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(91.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(55.0, m[:, 1].numpy().prod())


        m = Matrix.new([[3, 5], [7, 11]])
        m[0, :] = 3.0

        self.assertAlmostEqual(9.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(77.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(21.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(33.0, m[:, 1].numpy().prod())

        m = Matrix.new([[3, 5], [7, 11]])
        m[:, 0] = 3.0

        self.assertAlmostEqual(15.0, m[0, :].numpy().prod())
        self.assertAlmostEqual(33.0, m[1, :].numpy().prod())
        self.assertAlmostEqual(9.0, m[:, 0].numpy().prod())
        self.assertAlmostEqual(55.0, m[:, 1].numpy().prod())   

    def test_empty(self):
        m = Matrix()
        self.assertIsNotNone(m)
        self.assertTupleEqual((0, 0), m.size())

        with self.assertRaises(IndexError):
            m = Matrix.new([[]])

        with self.assertRaises(IndexError):
            m = Matrix.new([[], []])
        
    def test_nonempty(self):
        m = Matrix(100, 100)
        self.assertIsNotNone(m)
        self.assertTupleEqual((100, 100), m.size())

    def test_new(self):
        m = Matrix.new([[3, 5], [7, 11]])
        self.assertIsNotNone(m)
        self.assertTupleEqual((2, 2), m.size())
                
        m2 = Matrix.new(np.array([[3, 5], [7, 11]]))
        self.assertIsNotNone(m2)
        self.assertTupleEqual((2, 2), m2.size())
        
        self.assertTrue(m.equal(m2))
        self.assertTrue(m2.equal(m))

    def test__delitem__(self):
        m = Matrix()
        with self.assertRaises(IndexError):
            del m[0]

        m = Matrix.new([[3, 5], [7, 11]])

        del m[0] #deletes row 0

        self.assertTupleEqual((1, 2), m.size())
        self.assertAlmostEqual(7, m[0,0])
        self.assertAlmostEqual(11, m[0,1])

if __name__ == '__main__':
    unittest.main()
