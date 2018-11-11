from kaldi.base import math as kaldi_math
import math
import unittest


class testKaldiMath(unittest.TestCase):

    def testCrashes(self):
        """ Try to crash the interpreter """
        # Negative log?
        kaldi_math.log(-1.0)

        # Really small log
        kaldi_math.log(kaldi_math.DBL_EPSILON)

        # Negative rands
        with self.assertRaises(ValueError):
            kaldi_math.rand_int(-1, -50)
            kaldi_math.rand_int(10, -50)

        kaldi_math.rand_int(-100, 50)


    def testGcdLcmTpl(self):
        for a in range(1, 15):
            b = int(kaldi_math.rand() % 10)
            c = int(kaldi_math.rand() % 10)

            if kaldi_math.rand() % 2 == 0:
                b = -b
            if kaldi_math.rand() % 2 == 0:
                c = -c

            if b == 0 and c == 0:
                continue

            g = kaldi_math.gcd(b * a, c * a)

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
        self.assertEqual(1, kaldi_math.round_up_to_nearest_power_of_two(1))
        self.assertEqual(2, kaldi_math.round_up_to_nearest_power_of_two(2))
        self.assertEqual(4, kaldi_math.round_up_to_nearest_power_of_two(3))
        self.assertEqual(4, kaldi_math.round_up_to_nearest_power_of_two(4))
        self.assertEqual(8, kaldi_math.round_up_to_nearest_power_of_two(7))
        self.assertEqual(8, kaldi_math.round_up_to_nearest_power_of_two(8))
        self.assertEqual(256, kaldi_math.round_up_to_nearest_power_of_two(255))
        self.assertEqual(256, kaldi_math.round_up_to_nearest_power_of_two(256))
        self.assertEqual(512, kaldi_math.round_up_to_nearest_power_of_two(257))
        self.assertEqual(1073741824, kaldi_math.round_up_to_nearest_power_of_two(1073700000))

    def testDivideRoundingDown(self):
        for i in range(100):
            a = kaldi_math.rand_int(-100, 100)
            b = 0
            while b == 0:
                b = kaldi_math.rand_int(-100, 100)
            self.assertEqual(int(math.floor(float(a) / float(b))), kaldi_math.divide_rounding_down(a, b))

    @unittest.skip("TODO")
    def testRand(self):
        pass

    def testLogAddSub(self):
        for i in range(100):
            f1 = kaldi_math.rand() % 10000
            f2 = kaldi_math.rand() % 20
            add1 = kaldi_math.exp(kaldi_math.log_add(kaldi_math.log(f1), kaldi_math.log(f2)))
            add2 = kaldi_math.exp(kaldi_math.log_add(kaldi_math.log(f2), kaldi_math.log(f1)))
            add = f1 + f2
            thresh = add*0.00001

            self.assertAlmostEqual(add, add1, delta = thresh)
            self.assertAlmostEqual(add, add2, delta = thresh)

            try:
                f2_check = kaldi_math.exp(kaldi_math.log_sub(kaldi_math.log(add), kaldi_math.log(f1)))
                self.assertAlmostEqual(f2, f2_check, delta = thresh)
            except:
                # self.assertEqual(0, f2)
                pass

    def testDefines(self):
        self.assertAlmostEqual(0.0, kaldi_math.exp(kaldi_math.LOG_ZERO_FLOAT))
        self.assertAlmostEqual(0.0, kaldi_math.exp(kaldi_math.LOG_ZERO_DOUBLE))

        # TODO:
        # How to test these in Python?

        # den = 0.0
        # self.assertTrue(kaldi_math.KALDI_ISNAN(0.0 / den))
        # self.assertFalse(kaldi_math.KALDI_ISINF(0.0 / den))
        # self.assertFalse(kaldi_math.KALDI_ISFINITE(0.0 / den))
        # self.assertFalse(kaldi_math.KALDI_ISNAN(1.0 / den))
        # self.assertTrue(kaldi_math.KALDI_ISINF(1.0 / den))
        # self.assertFalse(kaldi_math.KALDI_ISFINITE(1.0 / den))
        # self.assertTrue(kaldi_math.KALDI_ISFINITE(0.0))
        # self.assertFalse(kaldi_math.KALDI_ISINF(0.0))
        # self.assertFalse(kaldi_math.KALDI_ISNAN(0.0))


        self.assertTrue(1.0 != 1.0 + kaldi_math.DBL_EPSILON)
        self.assertTrue(1.0 == 1.0 + 0.5 * kaldi_math.DBL_EPSILON)
        # self.assertNotAlmostEqual(1.0, 1.0 + kaldi_math.FLT_EPSILON, places = 7)
        # self.assertAlmostEqual(1.0, 1.0 + 0.5 * kaldi_math.FLT_EPSILON, places = 6)

        self.assertAlmostEqual(0.0, math.fabs(math.sin(kaldi_math.M_PI)))
        self.assertAlmostEqual(1.0, math.fabs(math.cos(kaldi_math.M_PI)))
        self.assertAlmostEqual(0.0, math.fabs(math.sin(kaldi_math.M_2PI)))
        self.assertAlmostEqual(1.0, math.fabs(math.cos(kaldi_math.M_2PI)))

        self.assertAlmostEqual(0.0, math.fabs(math.sin(kaldi_math.exp(kaldi_math.M_LOG_2PI))), places = 5)
        self.assertAlmostEqual(1.0, math.fabs(math.cos(kaldi_math.exp(kaldi_math.M_LOG_2PI))), places = 5)


if __name__ == '__main__':
    unittest.main()
