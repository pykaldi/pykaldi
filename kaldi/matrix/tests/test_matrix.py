from __future__ import division
import unittest
import numpy as np 
from kaldi.matrix import Matrix 

class TestKaldiMatrix(unittest.TestCase):

    def _test_str(self, m):
        # Test __str__ of kaldi.Matrix
        try:
            m.__str__()
        except:
            raise AssertionError("__str__ of Matrix failed")

    def test_empty(self):
        m = Matrix()
        self.assertIsNotNone(m)
        self.assertTrue(m.own_data)
        self.assertTupleEqual((0, 0), m.size())
        self._test_str(m)

        with self.raises(IndexError):
            m = Matrix.new([[]])
        # self.assertIsNotNone(m)
        # self.assertFalse(m.own_data)
        # self.assertTupleEqual((1, 0), m.size())
        # self._test_str(m)

        with self.raises(IndexError):
            m = Matrix.new([[], []])
        # self.assertIsNotNone(m)
        # self.assertFalse(m.own_data)
        # self.assertTupleEqual((2, 0), m.size())
        # self._test_str(m)

    def test_nonempty(self):
        m = Matrix(100, 100)
        self.assertIsNotNone(m)
        self.assertTrue(m.own_data)
        self.assertTupleEqual((100, 100), m.size())
        self._test_str(m)

    def test_new(self):
        m = Matrix.new([[3, 5], [7, 11]])
        self.assertIsNotNone(m)
        self.assertFalse(m.own_data)
        self.assertTupleEqual((2, 2), m.size())
        self._test_str(m)
                
        m2 = Matrix.new(np.array([[3, 5], [7, 11]]))
        self.assertIsNotNone(m2)
        self.assertFalse(m2.own_data)
        self.assertTupleEqual((2, 2), m2.size())
        self._test_str(m2)
        
        self.assertTrue(m.equal(m2))
        self.assertTrue(m2.equal(m))

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

    def test__delitem__(self):
        m = Matrix()
        with self.assertRaises(IndexError):
            del m[0]

        m = Matrix.new([[3, 5], [7, 11]])

        with self.assertRaises(ValueError):
            del m[0]

        m = m.clone()
        del m[0]
        self.assertTupleEqual((1, 2), m.size())
        self.assertAlmostEqual(7, m[0,0])
        self.assertAlmostEqual(11, m[0,1])

    def test_range(self):
        m = Matrix()

        self.assertTupleEqual((0, 0), m.range(0, 0, 0, 0).size())

        with self.assertRaises(IndexError):
            m.range(0, 1, 0, 0)

        with self.assertRaises(IndexError):
            m.range(0, 0, 0, 1)

        m = Matrix.new([[3, 5], [7, 11]])
        # TODO (VM):
        # Missing...


    def test_clone(self):
        m = Matrix()
        m2 = m.clone()
        self.assertTrue(m2.own_data)

        m = Matrix.new([[3, 5], [7, 11]])
        m2 = m.clone()
        self.assertTrue(m2.own_data)
        self.assertAlmostEqual(3.0, m2[0, 0])
        self.assertAlmostEqual(5.0, m2[0, 1])
        self.assertAlmostEqual(7.0, m2[1, 0])
        self.assertAlmostEqual(11.0, m2[1, 1])
        self._test_str(m2)

if __name__ == '__main__':
    unittest.main()
