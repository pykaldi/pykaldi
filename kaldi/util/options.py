import argparse
import sys

from . import _options_ext

class ParseOptions(_options_ext.ParseOptions):
    """Command line option parser.

    Args:
        usage (str): Usage string.
    """

    def parse_args(self, args=None):
        """Parses arguments.

        This method is used for parsing command line options. It fills the
        options objects registered with the parser. Parsed values for options
        that are directly registered with the parser, i.e. not via an options
        object, are returned as attributes of a `Namespace` object.

        Args:
            args (list): List of argument strings. If not provided, the argument
                strings are taken from `sys.argv`.

        Returns:
            A new `Namespace` object populated with the parsed values for
            options that are directly registered with the parser.
        """
        self._read(args if args else sys.argv)
        opts = self._get_options()
        arg_dict = {}
        arg_dict.update(opts.bool_map)
        arg_dict.update(opts.int_map)
        arg_dict.update(opts.uint_map)
        arg_dict.update(opts.float_map)
        arg_dict.update(opts.double_map)
        arg_dict.update(opts.str_map)
        return argparse.Namespace(**arg_dict)

################################################################################

_exclude_list = ['argparse', 'sys']

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')
           and not name in _exclude_list]
