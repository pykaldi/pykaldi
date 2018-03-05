from kaldi.util.table import *

import unittest

class TestKaldiTable(unittest.TestCase):
    def testClassifyWspecifier(self):
        a = "b,ark:foo|"
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("foo|", ark)
        self.assertEqual("", scp)
        self.assertTrue(opts.binary)

        a = "t,ark:foo|"
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("foo|", ark)
        self.assertEqual("", scp)
        self.assertFalse(opts.binary)

        a = "t,scp:a b c d"
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("", ark)
        self.assertEqual("a b c d", scp)
        self.assertFalse(opts.binary)

        a = "t,ark,scp:a b,c,d"
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.BOTH_SPECIFIER, ans)
        self.assertEqual("a b", ark)
        self.assertEqual("c,d", scp)
        self.assertFalse(opts.binary)

        a = ""
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.NO_SPECIFIER, ans)

        a = " t,ark:boo" #leading space not allowed.
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.NO_SPECIFIER, ans)

        a = " t,ark:boo"
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.NO_SPECIFIER, ans)

        a = "t,ark:boo " #trailing space not allowed.
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.NO_SPECIFIER, ans)

        a = "b,ark,scp:," #empty ark, scp fnames valid.
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.BOTH_SPECIFIER, ans)
        self.assertEqual("", ark)
        self.assertEqual("", scp)
        self.assertTrue(opts.binary)

        a = "f,b,ark,scp:," #empty ark, scp fnames valid.
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.BOTH_SPECIFIER, ans)
        self.assertEqual("", ark)
        self.assertEqual("", scp)
        self.assertTrue(opts.binary)
        self.assertTrue(opts.flush)

        a = "nf,b,ark,scp:," #empty ark, scp fnames valid.
        ans, ark, scp, opts = classify_wspecifier(a)
        self.assertEqual(WspecifierType.BOTH_SPECIFIER, ans)
        self.assertEqual("", ark)
        self.assertEqual("", scp)
        self.assertTrue(opts.binary)
        self.assertFalse(opts.flush)

    def testClassifyRspecifier(self):
        a = "ark:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("foo|", fname)

        a = "b,ark:foo|" #b, is ignored
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("foo|", fname)

        a = "ark,b:foo|" #,b is ignored
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("foo|", fname)

        a = "scp,b:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("foo|", fname)

        a = "scp,scp,b:foo|";  #invalid as repeated.
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)
        self.assertEqual("", fname)

        a = "ark,scp,b:foo|" #invalid as combined
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)
        self.assertEqual("", fname)

        a = "scp,o:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("foo|", fname)
        self.assertTrue(opts.once)

        a = "scp,no:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("foo|", fname)
        self.assertFalse(opts.once)

        a = "s,scp,no:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("foo|", fname)
        self.assertFalse(opts.once)
        self.assertTrue(opts.sorted)

        a = "scp:foo|"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("foo|", fname)

        a = "scp:" #empty fname valid
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("", fname)

        a = ""
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)

        a = "scp"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)

        a = "ark"
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)

        a = "ark:foo " #trailing space not allowed
        ans, fname, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.NO_SPECIFIER, ans)

        # Testing it accepts the meaningless t, and b, prefixes
        a = "b,scp:a"
        ans, b, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("a", b)

        a = "t,scp:a"
        ans, b, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.SCRIPT_SPECIFIER, ans)
        self.assertEqual("a", b)

        a = "b,ark:a"
        ans, b, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("a", b)

        a = "t,ark:a"
        ans, b, opts = classify_rspecifier(a)
        self.assertEqual(RspecifierType.ARCHIVE_SPECIFIER, ans)
        self.assertEqual("a", b)



if __name__ == '__main__':
    unittest.main()
