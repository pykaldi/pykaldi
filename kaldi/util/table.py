from . import _kaldi_table
from ._kaldi_table import ReadScriptFile, WriteScriptFile

################################################################################
# Sequential Readers
################################################################################

class _SequentialReaderBase(object):
    """Base class defining the additional Python API for sequential table
    readers.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key, value) pair from
    the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    def __init__(self, rspecifier=""):
        super(_SequentialReaderBase, self).__init__()
        if rspecifier != "":
            if not self.Open(rspecifier):
                raise IOError("Error opening SequentialTableReader with "
                              "rspecifier: {}".format(rspecifier))

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Returns the next (key, value) pair in the table.

        Returns:
            A (key, value) pair.

        Raises:
            StopIteration if there is no more items to return.
        """
        if self.Done():
            raise StopIteration
        else:
            key, value = self.Key(), self.Value()
            self.Next()
            return key, value


class SequentialVectorReader(_SequentialReaderBase,
                             _kaldi_table.SequentialVectorReader):
    """:kaldi:`kaldi::SequentialBaseFloatVectorReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :class:`~kaldi.matrix.Vector`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialMatrixReader(_SequentialReaderBase,
                             _kaldi_table.SequentialMatrixReader):
    """:kaldi:`kaldi::SequentialBaseFloatMatrixReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :class:`~kaldi.matrix.Matrix`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialWaveReader(_SequentialReaderBase,
                          _kaldi_table.SequentialWaveReader):
    """:kaldi:`kaldi::SequentialTableReader` <:kaldi:`kaldi::WaveHolder`>
    wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :class:`~kaldi.feat.WaveData`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialNnetExampleReader(_SequentialReaderBase,
                                  _kaldi_table.SequentialNnetExampleReader):
    """:kaldi:`kaldi::nnet3::SequentialNnetExampleReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :class:`~kaldi.nnet3.NnetExample`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialIntReader(_SequentialReaderBase,
                          _kaldi_table.SequentialIntReader):
    """:kaldi:`kaldi::SequentialInt32Reader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`int`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialFloatReader(_SequentialReaderBase,
                            _kaldi_table.SequentialFloatReader):
    """:kaldi:`kaldi::SequentialBaseFloatReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`float`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialDoubleReader(_SequentialReaderBase,
                             _kaldi_table.SequentialDoubleReader):
    """:kaldi:`kaldi::SequentialDoubleReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`float`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialBoolReader(_SequentialReaderBase,
                          _kaldi_table.SequentialBoolReader):
    """:kaldi:`kaldi::SequentialBoolReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`bool`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialIntVectorReader(_SequentialReaderBase,
                                _kaldi_table.SequentialIntVectorReader):
    """:kaldi:`kaldi::SequentialInt32VectorReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`list<int>`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialIntVectorVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialIntVectorVectorReader):
    """:kaldi:`kaldi::SequentialInt32VectorVectorReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`list<list<int>>`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialIntPairVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialIntPairVectorReader):
    """:kaldi:`kaldi::SequentialInt32PairVectorReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`list<(int,int)>`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class SequentialFloatPairVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialFloatPairVectorReader):
    """:kaldi:`kaldi::SequentialBaseFloatPairVectorReader` wrapper.

    This class implements the iterator protocol similar to how Python implements
    iteration over dictionaries. Each iteration returns a (key: :obj:`str`,
    value: :obj:`list<(float,float)>`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass

################################################################################
# Random Access Readers
################################################################################

class _RandomAccessReaderBase(object):
    """Base class defining the additional Python API for random access table
    readers.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    def __init__(self, rspecifier=""):
        super(_RandomAccessReaderBase, self).__init__()
        if rspecifier != "":
            if not self.Open(rspecifier):
                raise IOError("Error opening RandomAccessTableReader with "
                              "rspecifier: {}".format(rspecifier))

    def __enter__(self):
        return self

    def __contains__(self, key):
        return self.HasKey(key)

    def __getitem__(self, key):
        if self.HasKey(key):
            return self.Value(key)
        else:
            raise KeyError(key)


class RandomAccessVectorReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessVectorReader):
    """:kaldi:`kaldi::RandomAccessBaseFloatVectorReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessMatrixReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessMatrixReader):
    """:kaldi:`kaldi::RandomAccessBaseFloatMatrixReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessWaveReader(_RandomAccessReaderBase,
                             _kaldi_table.RandomAccessWaveReader):
    """:kaldi:`kaldi::RandomAccessTableReader` <:kaldi:`kaldi::WaveHolder`>
    wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessNnetExampleReader(_RandomAccessReaderBase,
                                    _kaldi_table.RandomAccessNnetExampleReader):
    """:kaldi:`kaldi::nnet3::RandomAccessNnetExampleReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessIntReader(_RandomAccessReaderBase,
                            _kaldi_table.RandomAccessIntReader):
    """:kaldi:`kaldi::RandomAccessInt32Reader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessFloatReader(_RandomAccessReaderBase,
                              _kaldi_table.RandomAccessFloatReader):
    """:kaldi:`kaldi::RandomAccessBaseFloatReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessDoubleReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessDoubleReader):
    """:kaldi:`kaldi::RandomAccessDoubleReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessBoolReader(_RandomAccessReaderBase,
                             _kaldi_table.RandomAccessBoolReader):
    """:kaldi:`kaldi::RandomAccessBoolReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessIntVectorReader(_RandomAccessReaderBase,
                                  _kaldi_table.RandomAccessIntVectorReader):
    """:kaldi:`kaldi::RandomAccessInt32VectorReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessIntVectorVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessIntVectorVectorReader):
    """:kaldi:`kaldi::RandomAccessInt32VectorVectorReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessIntPairVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessIntPairVectorReader):
    """:kaldi:`kaldi::RandomAccessInt32PairVectorReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass


class RandomAccessFloatPairVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessFloatPairVectorReader):
    """:kaldi:`kaldi::RandomAccessBaseFloatPairVectorReader` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.


    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError if opening the table for reading fails.
    """
    pass

################################################################################
# Random Access Mapped Readers
################################################################################

class _RandomAccessReaderMappedBase(object):
    """Base class defining the additional Python API for mapped random access
    table readers.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError if opening the table or map for reading fails.
    """
    def __init__(self, table_rspecifier="", map_rspecifier=""):
        super(_RandomAccessReaderMappedBase, self).__init__()
        if table_rspecifier != "" and map_rspecifier != "":
            if not self.Open(table_rspecifier, map_rspecifier):
                raise IOError("Error opening RandomAccessTableReaderMapped "
                              "with table_rspecifier: {}, map_rspecifier: {}"
                              .format(table_rspecifier, map_rspecifier))

    def __enter__(self):
        return self

    def __contains__(self, key):
        return self.HasKey(key)

    def __getitem__(self, key):
        if self.HasKey(key):
            return self.Value(key)
        else:
            raise KeyError(key)


class RandomAccessVectorReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessVectorReaderMapped):
    """:kaldi:`kaldi::RandomAccessBaseFloatVectorReaderMapped` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError if opening the table or map for reading fails.
    """
    pass


class RandomAccessMatrixReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessMatrixReaderMapped):
    """:kaldi:`kaldi::RandomAccessBaseFloatMatrixReaderMapped` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError if opening the table or map for reading fails.
    """
    pass


class RandomAccessFloatReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessFloatReaderMapped):
    """:kaldi:`kaldi::RandomAccessBaseFloatReaderMapped` wrapper.

    This class implements the __contains__ and __getitem__ methods to provide a
    dictionary-like API for accessing table entries.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError if opening the table or map for reading fails.
    """
    pass

################################################################################
# Writers
################################################################################

class _WriterBase(object):
    """Base class defining the additional Python API for table writers.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    def __init__(self, wspecifier=""):
        super(_WriterBase, self).__init__()
        if wspecifier != "":
            if not self.Open(wspecifier):
                raise IOError("Error opening TableWriter with wspecifier: {}"
                              .format(wspecifier))

    def __enter__(self):
        return self

    def __setitem__(self, key, value):
        self.Write(key, value)


class VectorWriter(_WriterBase, _kaldi_table.VectorWriter):
    """:kaldi:`kaldi::BaseFloatVectorWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class MatrixWriter(_WriterBase, _kaldi_table.MatrixWriter):
    """:kaldi:`kaldi::BaseFloatMatrixWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class WaveWriter(_WriterBase, _kaldi_table.WaveWriter):
    """:kaldi:`kaldi::TableWriter` <:kaldi:`kaldi::WaveHolder`> wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class NnetExampleWriter(_WriterBase, _kaldi_table.NnetExampleWriter):
    """:kaldi:`kaldi::nnet3::NnetExampleWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class IntWriter(_WriterBase, _kaldi_table.IntWriter):
    """:kaldi:`kaldi::Int32Writer` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class FloatWriter(_WriterBase, _kaldi_table.FloatWriter):
    """:kaldi:`kaldi::BaseFloatWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class DoubleWriter(_WriterBase, _kaldi_table.DoubleWriter):
    """:kaldi:`kaldi::DoubleWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class BoolWriter(_WriterBase, _kaldi_table.BoolWriter):
    """:kaldi:`kaldi::BoolWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class IntVectorWriter(_WriterBase, _kaldi_table.IntVectorWriter):
    """:kaldi:`kaldi::Int32VectorWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class IntVectorVectorWriter(_WriterBase, _kaldi_table.IntVectorVectorWriter):
    """:kaldi:`kaldi::Int32VectorVectorWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class IntPairVectorWriter(_WriterBase, _kaldi_table.IntPairVectorWriter):
    """:kaldi:`kaldi::Int32PairVectorWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass


class FloatPairVectorWriter(_WriterBase, _kaldi_table.FloatPairVectorWriter):
    """:kaldi:`kaldi::BaseFloatPairVectorWriter` wrapper.

    This class implements the __setitem__ method to provide a dictionary-like
    API for setting table entries.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError if opening the table for writing fails.
    """
    pass

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
