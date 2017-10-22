import os
from kaldi.base.math import *
from kaldi.util.io import *
from kaldi.base.io import read_line

import unittest

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

    def testIONew(self, binary = False):
        filename = "tmpf"
        ko = Output.new(filename, binary)
        outfile = ko.stream()
        ko.close()
        # ostream has no functions, use native python
        with open(filename, "w") as outpt:
            outpt.write("\t{}\t{}\n{}\t{}".format(500,
                                              600,
                                              700,
                                              "d")) #randomly selected char
        ki = Input()
        binary_contents = ki.open(filename)
        self.assertEqual(binary, binary_contents)

        # Read lines back
        self.assertEqual("\t500\t600", read_line(ki._input.stream()))

        for line in ki:
            self.assertEqual("700\td", line)


        os.remove(filename)

    
        

if __name__ == '__main__':
    unittest.main()
