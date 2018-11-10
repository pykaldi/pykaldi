import re
import os


import kaldi.util
from kaldi.matrix import Matrix
from kaldi.util.table import WaveWriter
from kaldi.feat.wave import WaveData

import numpy as np

################################################################################################################
# Auxiliary class that provides classname instanciation
################################################################################################################
class AuxMixin:
    def getClassname(self):
        """ Infers the name of the object to construct from the classname. """
        return self._test_replacer.sub('', self.__class__.__name__)

    def setUp(self):
        """ Sets up the inference for classname so that getImpl constructs the correct object """
        self._test_replacer = re.compile(re.escape("test"), re.IGNORECASE)
        self.classname = self.getClassname()
        self.filename = '/tmp/temp.ark'
        self.rspecifier = 'ark,t:{}'.format(self.filename)

    def getImpl(self, *args):
        """ Returns an instance of the self.classname class passing along the arguments for construction """
        if args:
            return getattr(kaldi.util.table, self.classname)(args[0])
        else:
            return getattr(kaldi.util.table, self.classname)()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)


################################################################################################################
# Examples mixins
################################################################################################################
class VectorExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one  [ 3 5 7 ]\n""" +\
                    """two  [ 1 2 3 ]\n""" +\
                    """three  [ ]\n""")
class MatrixExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one  [\n""" +\
                    """  0 1 2\n""" +\
                    """  3 4 5\n""" +\
                    """  6 7 8 ]\n""" +\
                    """two  [\n""" +\
                    """  1\n""" +\
                    """  2\n""" +\
                    """  3 ]\n""" +\
                    """three  [ ]\n""")

class WaveExampleMixin:
    def writeExample(self, outpt):
        m = Matrix(np.arange(9).reshape((3, 3)))
        with WaveWriter('ark:/tmp/temp.ark') as writer:
            writer['one'] = WaveData.from_data(1.0, m)

class IntExampleMixin:
    def writeExample(self, outpt):
        outpt.write("one 1\ntwo 2\nthree 3\n")

class FloatExampleMixin:
    def writeExample(self, outpt):
        outpt.write("one 1.0\ntwo 2.0\nthree 3.0\n")

class BoolExampleMixin:
    def writeExample(self, outpt):
        outpt.write("one T\ntwo T\nthree F\n")

class IntVectorExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one 1\n""" +\
                    """two 2 3\n""" +\
                    """three\n""")

class IntVectorVectorExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one 1 ;\n""" +\
                    """two 1 2 ; 3 4 ;\n""" +\
                    """three\n""")

class IntPairVectorExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one 1 1\n""" +\
                    """two 2 3 ; 4 5\n""" +\
                    """three\n""")

class FloatPairVectorExampleMixin:
    def writeExample(self, outpt):
        outpt.write("""one 1.0 1.0\n""" +\
                    """two 2.0 3.0 ; 4.0 5.0\n""" +\
                    """three\n""")
