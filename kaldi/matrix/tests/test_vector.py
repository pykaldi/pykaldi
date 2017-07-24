from __future__ import division
import unittest
import numpy as np 
from kaldi.matrix import Vector

class TestPyKaldi(unittest.TestCase):

    def _test_str(self, v):
        # Test __str__ of empty kaldi.Vector
        try:
            v.__str__()
        except:
            raise AssertionError("__str__ of empty Vector failed")

    def test_empty(self):

        # Test empty kaldi.Vector
        v = Vector()
        self.assertIsNotNone(v)
        self.assertTrue(v.own_data)
        self.assertEqual(0, v.size())
        self._test_str(v)

        v = Vector.new([])
        self.assertIsNotNone(v)
        self.assertFalse(v.own_data) #whyyyy???
        self.assertEqual(0, v.size())
        self._test_str(v)

    def test_nonempty(self):

        v = Vector(100)
        self.assertIsNotNone(v)
        self.assertTrue(v.own_data)
        self.assertEqual(100, v.size())
        self._test_str(v)

    def test_new(self):
        v = Vector.new([3, 5, 7, 11, 13])
        self.assertFalse(v.own_data)
        self.assertEqual(5, v.size())
        self.assertAlmostEqual(15015.0, v.numpy().prod())
        self._test_str(v)
        
        v2 = Vector.new(np.array([3, 5, 7, 11, 13]))
        self.assertFalse(v2.own_data)
        self.assertEqual(5, v2.size())
        self.assertAlmostEqual(15015.0, v2.numpy().prod())
        self._test_str(v2)

        self.assertTrue(v.equal(v2))
        self.assertTrue(v2.equal(v))

    def test__getitem__(self):
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
        v = Vector.new([3, 5, 7, 11, 13])
        v[0] = 15.0
        self.assertAlmostEqual(75075.0, v[:10].numpy().prod())

        with self.assertRaises(ValueError):
            v[0:3] = np.array([3, 5, 7, 11, 13])

        v[0:5] = np.array([3, 5, 7, 11, 13])
        self.assertAlmostEqual(15015.0, v.numpy().prod())

    def test__delitem__(self):
        v = Vector.new([3, 5, 7, 11, 13])

        # v does not own its data 
        with self.assertRaises(ValueError):
            del v[0]

        v = v.clone()
        del v[0]
        self.assertAlmostEqual(5005.0, v.numpy().prod())

    def test_range(self):
        v2 = Vector.new(np.array([3, 5, 7, 11, 13]))
        v3 = v2.range(1, 2)
        self.assertFalse(v3.own_data)
        self.assertTrue(2, v3.size())
        self.assertAlmostEqual(35.0, v3.numpy().prod())
        self._test_str(v3)

        self.assertTrue(np.all(v3.numpy() == v2[1:3].numpy()))

    def test_clone(self):
        v = Vector()
        v2 = v.clone()
        self.assertTrue(v2.own_data)

        v = Vector.new(np.array([3, 5, 7]))
        v2 = v.clone()
        self.assertTrue(v2.own_data)
        self.assertAlmostEqual(105, v2.numpy().prod())
        self._test_str(v2)

if __name__ == '__main__':
    unittest.main()
