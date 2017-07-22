from .kaldi_table import ReadScriptFile, WriteScriptFile
from . import kaldi_table_ext
from ..matrix import Vector, Matrix

################################################################################
# Sequential Readers
################################################################################

class SequentialVectorReader(kaldi_table_ext.SequentialVectorReader):
    """Python wrapper for
    kaldi::SequentialTableReader<KaldiObjectHolder<Vector<float>>>.

    This is a wrapper around the C extension type SequentialVectorReader.
    It provides a more Pythonic user facing API by implementing the iterator
    protocol.
    """

    def __init__(self, rspecifier=None):
        """Initializes a new sequential vector reader.

        If rspecifier is not None, also opens the specified table.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the data.
        """
        super(SequentialVectorReader, self).__init__()
        if rspecifier is not None:
            self.Open(rspecifier)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.Done():
            raise StopIteration
        else:
            k, v = self.Key(), Vector(src=self.Value())
            self.Next()
            return k, v


class SequentialMatrixReader(kaldi_table_ext.SequentialMatrixReader):
    """Python wrapper for
    kaldi::SequentialTableReader<KaldiObjectHolder<Matrix<float>>>.

    This is a wrapper around the C extension type SequentialMatrixReader.
    It provides a more Pythonic user facing API by implementing the iterator
    protocol.
    """

    def __init__(self, rspecifier=None):
        """Initializes a new sequential matrix reader.

        If rspecifier is not None, also opens the specified table.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the data.
        """
        super(SequentialMatrixReader, self).__init__()
        if rspecifier is not None:
            self.Open(rspecifier)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.Done():
            raise StopIteration
        else:
            k, v = self.Key(), Matrix(src=self.Value())
            self.Next()
            return k, v

################################################################################
# Random Access Readers
################################################################################

class RandomAccessVectorReader(kaldi_table_ext.RandomAccessVectorReader):
    """Python wrapper for
    kaldi::RandomAccessTableReader<KaldiObjectHolder<Vector<float>>>.

    This is a wrapper around the C extension type RandomAccessVectorReader.
    It provides a more Pythonic user facing API by implementing __contains__
    and __getitem__ methods.
    """

    def __init__(self, rspecifier=None):
        """Initializes a new random access vector reader.

        If rspecifier is not None, also opens the specified table.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the data.
        """
        super(RandomAccessVectorReader, self).__init__()
        if rspecifier is not None:
            self.Open(rspecifier)

    def __contains__(self, key):
        return self.HasKey(key)

    def __getitem__(self, key):
        if self.HasKey(key):
            return Vector(src=self.Value(key))
        else:
            raise KeyError(key)


class RandomAccessMatrixReader(kaldi_table_ext.RandomAccessMatrixReader):
    """Python wrapper for
    kaldi::RandomAccessTableReader<KaldiObjectHolder<Matrix<float>>>.

    This is a wrapper around the C extension type RandomAccessMatrixReader.
    It provides a more Pythonic user facing API by implementing __contains__
    and __getitem__ methods.
    """

    def __init__(self, rspecifier=None):
        """Initializes a new random access matrix reader.

        If rspecifier is not None, also opens the specified table.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the data.
        """
        super(RandomAccessMatrixReader, self).__init__()
        if rspecifier is not None:
            self.Open(rspecifier)

    def __contains__(self, key):
        return self.HasKey(key)

    def __getitem__(self, key):
        if self.HasKey(key):
            return Matrix(src=self.Value(key))
        else:
            raise KeyError(key)

################################################################################
# Writers
################################################################################

class VectorWriter(kaldi_table_ext.VectorWriter):
    """Python wrapper for
    kaldi::VectorWriter<KaldiObjectHolder<Vector<float>>>.

    This is a wrapper around the C extension type VectorWriter. It provides
    a more Pythonic user facing API by implementing __setitem__ method.
    """

    def __init__(self, wspecifier=None):
        """Initializes a new vector writer.

        If wspecifier is not None, also opens the specified table.

        Args:
            wspecifier(str): Kaldi wspecifier for writing the data.
        """
        super(VectorWriter, self).__init__()
        if wspecifier is not None:
            self.Open(wspecifier)

    def __setitem__(self, key, value):
        self.Write(key, value)


class MatrixWriter(kaldi_table_ext.MatrixWriter):
    """Python wrapper for
    kaldi::MatrixWriter<KaldiObjectHolder<Matrix<float>>>.

    This is a wrapper around the C extension type MatrixWriter. It provides
    a more Pythonic user facing API by implementing __setitem__ method.
    """

    def __init__(self, wspecifier=None):
        """Initializes a new matrix writer.

        If wspecifier is not None, also opens the specified table.

        Args:
            wspecifier(str): Kaldi wspecifier for writing the data.
        """
        super(MatrixWriter, self).__init__()
        if wspecifier is not None:
            self.Open(wspecifier)

    def __setitem__(self, key, value):
        self.Write(key, value)
