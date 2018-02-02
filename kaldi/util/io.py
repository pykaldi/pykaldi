"""
For detailed documentation of Kaldi input/output streams and extended filenames,
see `Kaldi I/O mechanisms`_ and `Kaldi I/O from a command-line perspective`_.

.. _Kaldi I/O mechanisms:
   http://kaldi-asr.org/doc/io.html
.. _Kaldi I/O from a command-line perspective:
   http://kaldi-asr.org/doc/io_tut.html
"""


from ..base import io as _base_io
from . import _kaldi_io
from ._kaldi_io import *


class Input(_kaldi_io.Input):
    """Input stream for reading from extended filenames.

    If **rxfilename** is provided, it is opened for reading.

    If **binary** is ``True``, the input stream is opened in binary mode.
    Otherwise, it is opened in text mode. If the stream is opened in binary mode
    and it has Kaldi binary mode header, `self.binary` attribute is set to
    ``True``. Similar to how files are handled in Python, PyKaldi distinguishes
    between input streams opened in binary and text modes, even when the
    underlying operating system doesn't. If the input stream is opened in binary
    mode, `read` and `readline` methods return contents as `bytes` objects
    without any decoding. In text mode, these methods return contents of the
    input stream as `unicode` strings, the bytes having been first decoded using
    the platform-dependent default encoding.

    This class implements the iterator and context manager protocols.

    Args:
        rxfilename (str): Extended filename to open for reading.
        binary (bool): Whether to open the stream in binary mode.

    Attributes:
        binary (bool): Whether the contents of the input stream are binary.
            This attribute is set to ``True`` if the stream is opened in binary
            mode and it has Kaldi binary mode header. Its value is valid only
            if the stream is still open.
    """

    def __init__(self, rxfilename=None, binary=True):
        super(Input, self).__init__()
        self.binary = False
        if rxfilename is not None:
            self.open(rxfilename, binary)

    def __del__(self):
        if self.is_open():
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.is_open():
            self.close()

    def __iter__(self):
        while True:
            line = self.readline()
            if not line:
                break
            yield line

    def open(self, rxfilename, binary=True):
        """Opens the stream for reading.

        Args:
            rxfilename (str): Extended filename to open for reading.
            binary (bool): Whether to open the stream in binary mode.
        """
        if binary:
            success, self.binary = super(Input, self).open(rxfilename)
        else:
            success = self._open_text_mode(rxfilename)
        if not success:
            raise IOError("Could not open stream for reading.")
        self._read = _base_io.read if binary else _base_io.read_text
        self._readline = _base_io.readline if binary else _base_io.readline_text

    def read(self):
        """Reads and returns the contents of the stream.

        If stream was opened in binary mode, returns a `bytes` object.
        Otherwise, returns a `unicode` string.
        """
        if not self.is_open():
            raise ValueError("I/O operation on closed stream.")
        return self._read(self.stream())

    def readline(self):
        """Reads and returns a line from the stream.

        If stream was opened in binary mode, returns a `bytes` object.
        Otherwise, returns a `unicode` string. If the stream is at EOF,
        an empty object is returned.
        """
        if not self.is_open():
            raise ValueError("I/O operation on closed stream.")
        return self._readline(self.stream())

    def readlines(self):
        """Reads and returns the contents of the stream as a list of lines.

        If stream was opened in binary mode, returns a list of `bytes` objects.
        Otherwise, returns a list of `unicode` strings.
        """
        return list(self)


class Output(_kaldi_io.Output):
    """Output stream for writing to extended filenames.

    If **wxfilename** is provided, it is opened for writing.

    If **binary** is ``True``, the output stream is opened in binary mode.
    Otherwise it is opened in text mode. Similar to how files are handled in
    Python, PyKaldi distinguishes between output streams opened in binary and
    text modes, even when the underlying operating system doesn't. If the output
    stream is opened in binary mode, `write` and `writelines` methods accept
    `bytes` objects. Otherwise, they accept `unicode` strings.

    If **write_header** is ``True`` and the stream was opened in binary mode,
    then Kaldi binary mode header (`\\\\0` then `B`) is written to the beginning
    of the stream. This header is checked by PyKaldi input streams opened in
    binary mode to set the `~Input.binary` attribute.

    This class implements the context manager protocol.

    Args:
        wxfilename (str): Extended filename to open for writing.
        binary (bool): Whether to open the stream in binary mode.
        write_header (bool): Whether to write Kaldi binary mode header in
            binary mode.
    """

    def __init__(self, wxfilename=None, binary=True, write_header=True):
        super(Output, self).__init__()
        if wxfilename is not None:
            self.open(wxfilename, binary, write_header)

    def __del__(self):
        if self.is_open():
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.is_open():
            self.close()

    def flush(self):
        """Flushes the stream."""
        if not self.is_open():
            raise ValueError("I/O operation on closed stream.")
        _base_io.flush(self.stream())

    def open(self, wxfilename, binary, write_header):
        """Opens the stream for writing.

        Args:
            wxfilename (str): Extended filename to open for writing.
            binary (bool): Whether to open the stream in binary mode.
            write_header (bool): Whether to write Kaldi binary mode header in
                binary mode.
        """
        if not super(Output, self).open(wxfilename, binary, write_header):
            raise IOError("Could not open stream for writing.")
        self._write = _base_io.write if binary else _base_io.write_text

    def write(self, s):
        """Writes s to the stream.

        Returns the number of bytes/characters written.
        """
        if not self.is_open():
            raise ValueError("I/O operation on closed stream.")
        if not self.stream().good():
            raise IOError("I/O error.")
        self._write(self.stream(), s)
        return len(s)

    def writelines(self, lines):
        """Writes a list of lines to the stream.

        Line separators are not added, so it is usual for each of the lines
        provided to have a line separator at the end.
        """
        for line in lines:
            self.write(line)


def xopen(xfilename, mode="r", write_header=True):
    """Opens an extended filename and returns the stream.

    The **mode** defaults to "r" which means open for reading in binary mode.
    The available modes are:

    ========= ===============================================================
    Character Meaning
    --------- ---------------------------------------------------------------
    'r'       open for reading (default)
    'w'       open for writing
    'b'       binary mode (default)
    't'       text mode
    ========= ===============================================================

    xopen() returns a stream object whose type depends on the mode, and
    through which the standard I/O operations such as reading and writing
    are performed. When xopen() is used to open a stream for reading ('r', 'rb',
    'rt'), it returns an `Input`. When used to open a stream for writing ('w',
    'wb', 'wt'), it return an `Output`.

    Similar to how files are handled in Python, PyKaldi distinguishes between
    streams opened in binary and text modes, even when the underlying operating
    system doesn't. If the stream is opened in binary mode, its I/O methods
    accept and return `bytes` objects. Otherwise, they accept and return
    `unicode` strings.

    Args:
        xfilename (str): Extended filename to open.
        mode (str): Optional string specifying the mode stream is opened.
        write_header (str): Whether streams opened for writing in binary mode
            write Kaldi binary mode header (`\\\\0` then `B`) to the beginning
            of the stream. This header is checked by streams opened for reading
            in binary mode to set `~Input.binary` attribute.
    """
    if not isinstance(xfilename, str):
        raise TypeError("invalid xfilename: %r" % xfilename)
    if not isinstance(mode, str):
        raise TypeError("invalid mode: %r" % mode)
    modes = set(mode)
    if modes - set("rwbt") or len(mode) > len(modes):
        raise ValueError("invalid mode: %r" % mode)
    reading = "r" in modes
    writing = "w" in modes
    text = "t" in modes
    binary = "b" in modes
    if text and binary:
        raise ValueError("can't have text and binary mode at once")
    if reading + writing != 1:
        raise ValueError("must have exactly one of read/write mode")
    if not text:
        binary = True
    if reading:
        return Input(xfilename, binary)
    else:
        return Output(xfilename, binary, write_header)


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
