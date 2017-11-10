from . import _kaldi_table
from ._kaldi_table import (read_script_file, write_script_file,
                           classify_wspecifier, classify_rspecifier,
                           WspecifierType, RspecifierType)

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
            if not self.open(rspecifier):
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
        if self.done():
            raise StopIteration
        else:
            key, value = self.key(), self.value()
            self._next()
            return key, value


class SequentialVectorReader(_SequentialReaderBase,
                             _kaldi_table.SequentialVectorReader):
    """Sequential table reader for single precision vectors.

    This class is used for reading single precision vectors sequentially from an
    archive or script file. It implements the iterator protocol similar to how
    Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :class:`~kaldi.matrix.Vector`) pair from the table
    in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialMatrixReader(_SequentialReaderBase,
                             _kaldi_table.SequentialMatrixReader):
    """Sequential table reader for single precision matrices.

    This class is used for reading single precision matrices sequentially from
    an archive or script file. It implements the iterator protocol similar to
    how Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :class:`~kaldi.matrix.Matrix`) pair from the table
    in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialWaveReader(_SequentialReaderBase,
                          _kaldi_table.SequentialWaveReader):
    """Sequential table reader for wave files.

    This class is used for reading wave files sequentially from an archive or
    script file. It implements the iterator protocol similar to how Python
    implements iteration over dictionaries. Each iteration returns a (key:
    :obj:`str`, value: :class:`~kaldi.feat.wave.WaveData`) pair from the table
    in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialNnetExampleReader(_SequentialReaderBase,
                                  _kaldi_table.SequentialNnetExampleReader):
    """Sequential table reader for nnet examples.

    This class is used for reading nnet examples sequentially from an archive or
    script file. It implements the iterator protocol similar to how Python
    implements iteration over dictionaries. Each iteration returns a (key:
    :obj:`str`, value: :class:`~kaldi.nnet3.NnetExample`) pair from the table in
    sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialIntReader(_SequentialReaderBase,
                          _kaldi_table.SequentialIntReader):
    """Sequential table reader for integers.

    This class is used for reading integers sequentially from an archive or
    script file. It implements the iterator protocol similar to how Python
    implements iteration over dictionaries. Each iteration returns a (key:
    :obj:`str`, value: :obj:`int`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass

class SequentialFloatReader(_SequentialReaderBase,
                            _kaldi_table.SequentialFloatReader):
    """Sequential table reader for single precision floats.

    This class is used for reading single precision floats sequentially from an
    archive or script file. It implements the iterator protocol similar to how
    Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :obj:`float`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialDoubleReader(_SequentialReaderBase,
                             _kaldi_table.SequentialDoubleReader):
    """Sequential table reader for double precision floats.

    This class is used for reading double precision floats sequentially from an
    archive or script file. It implements the iterator protocol similar to how
    Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :obj:`float`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialBoolReader(_SequentialReaderBase,
                          _kaldi_table.SequentialBoolReader):
    """Sequential table reader for Booleans.

    This class is used for reading Booleans sequentially from an archive or
    script file. It implements the iterator protocol similar to how Python
    implements iteration over dictionaries. Each iteration returns a (key:
    :obj:`str`, value: :obj:`bool`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialIntVectorReader(_SequentialReaderBase,
                                _kaldi_table.SequentialIntVectorReader):
    """Sequential table reader for integer sequences.

    This class is used for reading integer sequences sequentially from an
    archive or script file. It implements the iterator protocol similar to how
    Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :obj:`List[int]`) pair from the table in sequential
    order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialIntVectorVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialIntVectorVectorReader):
    """Sequential table reader for sequences of integer sequences.

    This class is used for reading sequences of integer sequences sequentially
    from an archive or script file. It implements the iterator protocol similar
    to how Python implements iteration over dictionaries. Each iteration returns
    a (key: :obj:`str`, value: :obj:`List[List[int]]`) pair from the table in
    sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialIntPairVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialIntPairVectorReader):
    """Sequential table reader for sequences of integer pairs.

    This class is used for reading sequences of integer pairs sequentially from
    an archive or script file. It implements the iterator protocol similar to
    how Python implements iteration over dictionaries. Each iteration returns a
    (key: :obj:`str`, value: :obj:`List[Tuple[int,int]]`) pair from the table in
    sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class SequentialFloatPairVectorReader(
        _SequentialReaderBase,
        _kaldi_table.SequentialFloatPairVectorReader):
    """Sequential table reader for sequences of single precision float pairs.

    This class is used for reading sequences of single precision float pairs
    sequentially from an archive or script file. It implements the iterator
    protocol similar to how Python implements iteration over dictionaries. Each
    iteration returns a (key: :obj:`str`, value:
    :obj:`List[Tuple[float,float]]`) pair from the table in sequential order.

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
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
            if not self.open(rspecifier):
                raise IOError("Error opening RandomAccessTableReader with "
                              "rspecifier: {}".format(rspecifier))

    def __enter__(self):
        return self

    def __contains__(self, key):
        return self.has_key(key)

    def __getitem__(self, key):
        if self.has_key(key):
            return self.value(key)
        else:
            raise KeyError(key)


class RandomAccessVectorReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessVectorReader):
    """Random access table reader for single precision vectors.

    This class is used for randomly accessing single precision vectors in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (a :class:`~kaldi.matrix.Vector`)
    associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessMatrixReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessMatrixReader):
    """Random access table reader for single precision matrices.

    This class is used for randomly accessing single precision matrices in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (a :class:`~kaldi.matrix.Matrix`)
    associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessWaveReader(_RandomAccessReaderBase,
                             _kaldi_table.RandomAccessWaveReader):
    """Random access table reader for wave files.

    This class is used for randomly accessing wave files in an archive or script
    file. It implements `__contains__` and `__getitem__` methods to provide a
    dictionary-like interface for accessing table entries. e.g. `reader[key]`
    returns the item (a :class:`~kaldi.feat.wave.WaveData`) associated with the
    key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessNnetExampleReader(_RandomAccessReaderBase,
                                    _kaldi_table.RandomAccessNnetExampleReader):
    """Random access table reader for nnet examples.

    This class is used for randomly accessing nnet examples in an archive or
    script file. It implements `__contains__` and `__getitem__` methods to
    provide a dictionary-like interface for accessing table entries. e.g.
    `reader[key]` returns the item (a :class:`~kaldi.nnet3.NnetExample`)
    associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessIntReader(_RandomAccessReaderBase,
                            _kaldi_table.RandomAccessIntReader):
    """Random access table reader for integers.

    This class is used for randomly accessing integers in an archive or script
    file. It implements `__contains__` and `__getitem__` methods to provide a
    dictionary-like interface for accessing table entries. e.g. `reader[key]`
    returns the item (an :obj:`int`) associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessFloatReader(_RandomAccessReaderBase,
                              _kaldi_table.RandomAccessFloatReader):
    """Random access table reader for single precision floats.

    This class is used for randomly accessing single precision floats in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (a :obj:`float`) associated with the key
    (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessDoubleReader(_RandomAccessReaderBase,
                               _kaldi_table.RandomAccessDoubleReader):
    """Random access table reader for double precision floats.

    This class is used for randomly accessing double precision floats in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (a :obj:`float`) associated with the key
    (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessBoolReader(_RandomAccessReaderBase,
                             _kaldi_table.RandomAccessBoolReader):
    """Random access table reader for Booleans.

    This class is used for randomly accessing Booleans in an archive or script
    file. It implements `__contains__` and `__getitem__` methods to provide a
    dictionary-like interface for accessing table entries. e.g. `reader[key]`
    returns the item (an :obj:`bool`) associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessIntVectorReader(_RandomAccessReaderBase,
                                  _kaldi_table.RandomAccessIntVectorReader):
    """Random access table reader for integer sequences.

    This class is used for randomly accessing integer sequences in an archive or
    script file. It implements `__contains__` and `__getitem__` methods to
    provide a dictionary-like interface for accessing table entries. e.g.
    `reader[key]` returns the item (an :obj:`List[int]`) associated with the key
    (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessIntVectorVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessIntVectorVectorReader):
    """Random access table reader for sequences of integer sequences.

    This class is used for randomly accessing sequences of integer sequences in
    an archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (an :obj:`List[List[int]]`) associated
    with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessIntPairVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessIntPairVectorReader):
    """Random access table reader for sequences of integer pairs.

    This class is used for randomly accessing sequences of integer pairs in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    e.g. `reader[key]` returns the item (an :obj:`List[Tuple[int, int]]`)
    associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
    """
    pass


class RandomAccessFloatPairVectorReader(
        _RandomAccessReaderBase,
        _kaldi_table.RandomAccessFloatPairVectorReader):
    """Random access table reader for sequences of single precision float pairs.

    This class is used for randomly accessing sequences of single precision
    float pairs in an archive or script file. It implements `__contains__` and
    `__getitem__` methods to provide a dictionary-like interface for accessing
    table entries. e.g. `reader[key]` returns the item (an
    :obj:`List[Tuple[float, float]]`) associated with the key (a :obj:`str`).

    Args:
        rspecifier(str): Kaldi rspecifier for reading the table. If provided,
            the table is opened for reading.

    Raises:
        IOError: If opening the table for reading fails.
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
        return self.has_key(key)

    def __getitem__(self, key):
        if self.has_key(key):
            return self.value(key)
        else:
            raise KeyError(key)


class RandomAccessVectorReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessVectorReaderMapped):
    """Mapped random access table reader for single precision vectors.

    This class is used for randomly accessing single precision vectors in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    If a **map_rspecifier** is provided, the map is used for converting the keys
    to the actual keys used to query the table, e.g. `reader[key]` returns the
    item (a :class:`~kaldi.matrix.Vector`) associated with the key `map[key]`
    (a :obj:`str`). Otherwise, it works like :class:`RandomAccessVectorReader`.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError: If opening the table or map for reading fails.
    """
    pass


class RandomAccessMatrixReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessMatrixReaderMapped):
    """Mapped random access table reader for single precision matrices.

    This class is used for randomly accessing single precision matrices in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    If a **map_rspecifier** is provided, the map is used for converting the keys
    to the actual keys used to query the table, e.g. `reader[key]` returns the
    item (a :class:`~kaldi.matrix.Matrix`) associated with the key `map[key]`
    (a :obj:`str`). Otherwise, it works like :class:`RandomAccessMatrixReader`.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError: If opening the table or map for reading fails.
    """
    pass


class RandomAccessFloatReaderMapped(
        _RandomAccessReaderMappedBase,
        _kaldi_table.RandomAccessFloatReaderMapped):
    """Mapped random access table reader for single precision floats.

    This class is used for randomly accessing single precision floats in an
    archive or script file. It implements `__contains__` and `__getitem__`
    methods to provide a dictionary-like interface for accessing table entries.
    If a **map_rspecifier** is provided, the map is used for converting the keys
    to the actual keys used to query the table, e.g. `reader[key]` returns the
    item (a :obj:`float`) associated with the key `map[key]` (a :obj:`str`).
    Otherwise, it works like :class:`RandomAccessFloatReader`.

    Args:
        table_rspecifier(str): Kaldi rspecifier for reading the table.
            If provided, the table is opened for reading.
        map_rspecifier (str): Kaldi rspecifier for reading the map.
            If provided, the map is opened for reading.

    Raises:
        IOError: If opening the table or map for reading fails.
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
            if not self.open(wspecifier):
                raise IOError("Error opening TableWriter with wspecifier: {}"
                              .format(wspecifier))

    def __enter__(self):
        return self

    def __setitem__(self, key, value):
        self.write(key, value)


class VectorWriter(_WriterBase, _kaldi_table.VectorWriter):
    """Table writer for single precision vectors.

    This class is used for writing single precision vectors to an archive or
    script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value:
    :class:`~kaldi.matrix.Vector`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class MatrixWriter(_WriterBase, _kaldi_table.MatrixWriter):
    """Table writer for single precision matrices.

    This class is used for writing single precision matrices to an archive or
    script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value:
    :class:`~kaldi.matrix.Matrix`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class WaveWriter(_WriterBase, _kaldi_table.WaveWriter):
    """Table writer for wave files.

    This class is used for writing wave files to an archive or script file. It
    implements the `__setitem__` method to provide a dictionary-like interface
    for writing table entries, e.g. `writer[key] = value` writes the pair (key:
    :obj:`str`, value: :class:`~kaldi.feat.wave.WaveData`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class NnetExampleWriter(_WriterBase, _kaldi_table.NnetExampleWriter):
    """Table writer for nnet examples.

    This class is used for writing nnet examples to an archive or script file.
    It implements the `__setitem__` method to provide a dictionary-like
    interface for writing table entries, e.g. `writer[key] = value` writes the
    pair (key: :obj:`str`, value: :class:`~kaldi.nnet3.NnetExample`) pair to the
    table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class IntWriter(_WriterBase, _kaldi_table.IntWriter):
    """Table writer for integers.

    This class is used for writing integers to an archive or script file. It
    implements the `__setitem__` method to provide a dictionary-like interface
    for writing table entries, e.g. `writer[key] = value` writes the pair (key:
    :obj:`str`, value: :obj:`int`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class FloatWriter(_WriterBase, _kaldi_table.FloatWriter):
    """Table writer for single precision floats.

    This class is used for writing single precision floats to an archive or
    script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value: :obj:`float`) pair to the
    table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class DoubleWriter(_WriterBase, _kaldi_table.DoubleWriter):
    """Table writer for double precision floats.

    This class is used for writing double precision floats to an archive or
    script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value: :obj:`float`) pair to the
    table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class BoolWriter(_WriterBase, _kaldi_table.BoolWriter):
    """Table writer for Booleans.

    This class is used for writing Booleans to an archive or script file. It
    implements the `__setitem__` method to provide a dictionary-like interface
    for writing table entries, e.g. `writer[key] = value` writes the pair (key:
    :obj:`str`, value: :obj:`bool`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class IntVectorWriter(_WriterBase, _kaldi_table.IntVectorWriter):
    """Table writer for integer sequences.

    This class is used for writing integer sequences to an archive or script
    file. It implements the `__setitem__` method to provide a dictionary-like
    interface for writing table entries, e.g. `writer[key] = value` writes the
    pair (key: :obj:`str`, value: :obj:`List[int]`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class IntVectorVectorWriter(_WriterBase, _kaldi_table.IntVectorVectorWriter):
    """Table writer for sequences of integer sequences.

    This class is used for writing sequences of integer sequences to an archive
    or script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value: :obj:`List[List[int]]`) pair
    to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class IntPairVectorWriter(_WriterBase, _kaldi_table.IntPairVectorWriter):
    """Table writer for sequences of integer pairs.

    This class is used for writing sequences of integer pairs to an archive or
    script file. It implements the `__setitem__` method to provide a
    dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value: :obj:`List[Tuple[int,int]]`)
    pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass


class FloatPairVectorWriter(_WriterBase, _kaldi_table.FloatPairVectorWriter):
    """Table writer for sequences of single precision float pairs.

    This class is used for writing sequences of single precision float pairs to
    an archive or script file. It implements the `__setitem__` method to provide
    a dictionary-like interface for writing table entries, e.g. `writer[key] =
    value` writes the pair (key: :obj:`str`, value:
    :obj:`List[Tuple[float,float]]`) pair to the table.

    Args:
        wspecifier (str): Kaldi wspecifier for writing the table. If provided,
            the table is opened for writing.

    Raises:
        IOError: If opening the table for writing fails.
    """
    pass

################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
