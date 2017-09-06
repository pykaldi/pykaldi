from kaldi.base import kaldi_math 
import math
import unittest

class testKaldiMath(unittest.TestCase):

	def testGcdLcmTpl(self):
		for a in range(1, 15):
			b = int(kaldi_math.Rand() % 10)
			c = int(kaldi_math.Rand() % 10)

			if kaldi_math.Rand() % 2 == 0:
				b = -b
			if kaldi_math.Rand() % 2 == 0:
				c = -c 

			if b == 0 and c == 0:
				continue 

			g = kaldi_math.Gcd(b * a, c * a)

			self.assertTrue(g >= a)
			self.assertEqual(0, (b * a) % g)
			self.assertEqual(0, (c * a) % g)

			if b <= 0 or c <= 0:
				with self.assertRaises(ValueError):
					kaldi_math.lcm(b * a, c * a)
			else:
				h = kaldi_math.lcm(b * a, c * a)
				self.assertNotEqual(0, h)
				self.assertEqual(0, (h % (b * a)))
				self.assertEqual(0, (h % (c * a)))

	def testRoundUpToNearestPowerOfTwo(self):
		self.assertEqual(1, kaldi_math.RoundUpToNearestPowerOfTwo(1))
		self.assertEqual(2, kaldi_math.RoundUpToNearestPowerOfTwo(2))
		self.assertEqual(4, kaldi_math.RoundUpToNearestPowerOfTwo(3))
		self.assertEqual(4, kaldi_math.RoundUpToNearestPowerOfTwo(4))
		self.assertEqual(8, kaldi_math.RoundUpToNearestPowerOfTwo(7))
		self.assertEqual(8, kaldi_math.RoundUpToNearestPowerOfTwo(8))
		self.assertEqual(256, kaldi_math.RoundUpToNearestPowerOfTwo(255))
		self.assertEqual(256, kaldi_math.RoundUpToNearestPowerOfTwo(256))
		self.assertEqual(512, kaldi_math.RoundUpToNearestPowerOfTwo(257))
		self.assertEqual(1073741824, kaldi_math.RoundUpToNearestPowerOfTwo(1073700000))

	def testDivideRoundingDown(self):
		for i in range(100):
			a = kaldi_math.RandInt(-100, 100)
			b = 0
			while b == 0:
				b = kaldi_math.RandInt(-100, 100)
				self.assertEqual(int(math.floor(float(a) / float(b))), kaldi_math.DivideRoundingDown(a, b))

	@unittest.skip("TODO")
	def testRand(self):
		pass		

	def testLogAddSub(self):
		for i in range(100):
			f1 = kaldi_math.Rand() % 10000
			f2 = kaldi_math.Rand() % 20
			add1 = kaldi_math.Exp(kaldi_math.LogAdd(kaldi_math.Log(f1), kaldi_math.Log(f2)))
			add2 = kaldi_math.Exp(kaldi_math.LogAdd(kaldi_math.Log(f2), kaldi_math.Log(f1)))
			add = f1 + f2 
			thresh = add*0.00001

			self.assertAlmostEqual(add, add1, delta = thresh)
			self.assertAlmostEqual(add, add2, delta = thresh)

			try:
				f2_check = kaldi_math.Exp(kaldi_math.LogSub(kaldi_math.Log(add), kaldi_math.Log(f1)))
				self.assertAlmostEqual(f2, f2_check, delta = thresh)
			except:
				# self.assertEqual(0, f2)
				pass

	def testDefines(self):
		self.assertAlmostEqual(0.0, kaldi_math.Exp(kaldi_math.kLogZeroFloat))
		self.assertAlmostEqual(0.0, kaldi_math.Exp(kaldi_math.kLogZeroDouble))

		den = 0.0
		self.assertTrue(kaldi_math.KALDI_ISNAN(0.0 / den))
		self.assertFalse(kaldi_math.KALDI_ISINF(0.0 / den))
		self.assertFalse(kaldi_math.KALDI_ISFINITE(0.0 / den))
		self.assertFalse(kaldi_math.KALDI_ISNAN(1.0 / den))
		self.assertTrue(kaldi_math.KALDI_ISINF(1.0 / den))
		self.assertFalse(kaldi_math.KALDI_ISFINITE(1.0 / den))
		self.assertTrue(kaldi_math.KALDI_ISFINITE(0.0))
		self.assertFalse(kaldi_math.KALDI_ISINF(0.0))
		self.assertFalse(kaldi_math.KALDI_ISNAN(0.0))

		self.assertNotEqual(1.0, 1.0 + DBL_EPSILON)
		self.assertEqual(1.0, 1.0 + 0.5 * DBL_EPSILON)
		self.assertNotEqual(1.0, 1.0 + FLT_EPSILON)
		self.assertEqual(1.0, 1.0 + 0.5 * FLT_EPSILON)
		
		self.assertAlmostEqual(0.0, math.fabs(math.sin(M_PI)))
		self.assertAlmostEqual(-1.0, math.fabs(math.cos(M_PI)))
		self.assertAlmostEqual(0.0, math.fabs(math.sin(M_2PI)))
		self.assertAlmostEqual(1.0, math.fabs(math.cos(M_2PI)))

		self.assertAlmostEqual(0.0, math.fabs(math.sin(kaldi_math.Exp(M_LOG_2PI))))
		self.assertAlmostEqual(1.0, math.fabs(math.cos(kaldi_math.Exp(M_LOG_2PI))))


if __name__ == '__main__':
    unittest.main()
