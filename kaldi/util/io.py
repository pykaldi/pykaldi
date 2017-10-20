from . import _kaldi_io
from kaldi.base.io import read_line

################################################################################
# 
################################################################################
class Input(object):
    """:kaldi:`kaldi::Input` wrapper.

    This class implements iterable and context manager.
    """
    def __init__(self, rspecifier = ""):
        self._input = _kaldi_io._Input()
        if rspecifier != "":
            self._input.open(rspecifier)

    def open(self, rspecifier):
        return self._input.open(rspecifier)

    def open_text_mode(self, rspecifier):
        return self._input.open_text_mode(rspecifier)

    def is_open(self):
        return self._input.is_open()

    def close(self):
        return self._input.close()

    def __iter__(self):
        s = read_line(self._input.stream())
        while(s):
            yield s
            s = read_line(self._input.stream())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._input.close()

################################################################################

_exclude_list = ['istream', 'ostream', 'iostream']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
