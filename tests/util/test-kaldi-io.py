from __future__ import print_function

import os
import unittest

from kaldi.util.io import *


class TestKaldiIO(unittest.TestCase):

    def testClassifyRxfilename(self):
        self.assertEqual(InputType.STANDARD_INPUT, classify_rxfilename(""))
        self.assertEqual(InputType.NO_INPUT, classify_rxfilename(" "))
        self.assertEqual(InputType.NO_INPUT, classify_rxfilename(" a "))
        self.assertEqual(InputType.NO_INPUT, classify_rxfilename("a "))
        self.assertEqual(InputType.FILE_INPUT, classify_rxfilename("a"))
        self.assertEqual(InputType.STANDARD_INPUT, classify_rxfilename("-"))
        self.assertEqual(InputType.PIPE_INPUT, classify_rxfilename("b|"))
        self.assertEqual(InputType.NO_INPUT, classify_rxfilename("|b"))
        self.assertEqual(InputType.PIPE_INPUT, classify_rxfilename("b c|"))
        self.assertEqual(InputType.OFFSET_FILE_INPUT, classify_rxfilename("a b c:123"))
        self.assertEqual(InputType.OFFSET_FILE_INPUT, classify_rxfilename("a b c:3"))
        self.assertEqual(InputType.FILE_INPUT, classify_rxfilename("a b c:"))
        self.assertEqual(InputType.FILE_INPUT, classify_rxfilename("a b c/3"))

    def testClassifyWxfilename(self):
        self.assertEqual(OutputType.STANDARD_OUTPUT, classify_wxfilename(""))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename(" "))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename(" a "))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename("a "))
        self.assertEqual(OutputType.FILE_OUTPUT, classify_wxfilename("a"))
        self.assertEqual(OutputType.STANDARD_OUTPUT, classify_wxfilename("-"))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename("b|"))
        self.assertEqual(OutputType.PIPE_OUTPUT, classify_wxfilename("|b"))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename("b c|"))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename("a b c:123"))
        self.assertEqual(OutputType.NO_OUTPUT, classify_wxfilename("a b c:3"))
        self.assertEqual(OutputType.FILE_OUTPUT, classify_wxfilename("a b c:"))
        self.assertEqual(OutputType.FILE_OUTPUT, classify_wxfilename("a b c/3"))

    def test_text_io(self):
        filename = "tmpf"
        lines = ["400\t500\t600", "700\td"]
        with Output(filename, False) as ko:
            for line in lines:
                print(line, file=ko)
        with Input(filename, False) as ki:
            for i, line in enumerate(ki):
                self.assertEqual(line.strip(), lines[i])
        os.remove(filename)

    def test_binary_io(self):
        filename = "tmpf"
        lines = [b"\t500\t600\n", b"700\td\n"]
        with Output(filename) as ko:
            for line in lines:
                ko.write(line)
        with Input(filename) as ki:
            self.assertTrue(ki.binary)
            for i, line in enumerate(ki):
                self.assertEqual(line, lines[i])
        os.remove(filename)

    def test_xopen(self):
        filename = "tmpf"
        lines = [b"\t500\t600\n", b"700\td\n"]
        with xopen(filename, "w") as ko:
            ko.writelines(lines)
        with xopen(filename) as ki:
            self.assertTrue(ki.binary)
            for i, line in enumerate(ki):
                self.assertEqual(line, lines[i])
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
