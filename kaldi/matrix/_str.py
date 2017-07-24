# Adapted from pytorch tensor printing
# https://github.com/pytorch/pytorch/blob/master/torch/_tensor_str.py

import math
import numpy
from functools import reduce


class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80


PRINT_OPTS = __PrinterOptions()
SCALE_FORMAT = '{:.5e} *\n'


# We could use **kwargs, but this will give better docs
def set_printoptions(
        precision=None,
        threshold=None,
        edgeitems=None,
        linewidth=None,
        profile=None,
):
    """Set options for printing. Items shamelessly taken from Numpy

    Args:
        precision: Number of digits of precision for floating point output
            (default 8).
        threshold: Total number of array elements which trigger summarization
            rather than full repr (default 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default 80). Thresholded matricies will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (default, short, full)
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = float('inf')
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth


def _range(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)


def _number_format(self, min_sz=-1):
    min_sz = max(min_sz, 2)
    temp = numpy.abs(self.reshape(self.size), dtype=float)

    invalid_value_mask = ~numpy.isfinite(temp)
    if invalid_value_mask.all():
        example_value = 0
    else:
        example_value = temp[invalid_value_mask == 0][0]
    temp[invalid_value_mask] = example_value
    if invalid_value_mask.any():
        min_sz = max(min_sz, 3)

    int_mode = True
    # TODO: use fmod?
    for value in temp:
        if value != math.ceil(value):
            int_mode = False
            break

    exp_min = temp.min()
    if exp_min != 0:
        exp_min = math.floor(math.log10(exp_min)) + 1
    else:
        exp_min = 1
    exp_max = temp.max()
    if exp_max != 0:
        exp_max = math.floor(math.log10(exp_max)) + 1
    else:
        exp_max = 1

    scale = 1
    exp_max = int(exp_max)
    prec = PRINT_OPTS.precision
    if int_mode:
        if exp_max > prec + 1:
            format = '{{:11.{}e}}'.format(prec)
            sz = max(min_sz, 7 + prec)
        else:
            sz = max(min_sz, exp_max + 1)
            format = '{:' + str(sz) + '.0f}'
    else:
        if exp_max - exp_min > prec:
            sz = 7 + prec
            if abs(exp_max) > 99 or abs(exp_min) > 99:
                sz = sz + 1
            sz = max(min_sz, sz)
            format = '{{:{}.{}e}}'.format(sz, prec)
        else:
            if exp_max > prec + 1 or exp_max < 0:
                sz = max(min_sz, 7)
                scale = math.pow(10, exp_max - 1)
            else:
                if exp_max == 0:
                    sz = 7
                else:
                    sz = exp_max + 6
                sz = max(min_sz, sz)
            format = '{{:{}.{}f}}'.format(sz, prec)
    return format, scale, sz


def __repr_row(row, indent, fmt, scale, sz, truncate=None):
    if truncate is not None:
        dotfmt = " {:^5} "
        return (indent +
                ' '.join(fmt.format(val / scale) for val in row[:truncate]) +
                dotfmt.format('...') +
                ' '.join(fmt.format(val / scale) for val in row[-truncate:]) +
                '\n')
    else:
        return indent + ' '.join(fmt.format(val / scale) for val in row) + '\n'


def _matrix_str(self, indent='', formatter=None, force_truncate=False):
    type_str = self.__module__ + '.' + self.__class__.__name__
    self = self.numpy()
    if self.size == 0:
        return '[{} with no elements]\n'.format(type_str)
    n = PRINT_OPTS.edgeitems
    has_hdots = self.shape[1] > 2 * n
    has_vdots = self.shape[0] > 2 * n
    print_full_mat = not has_hdots and not has_vdots

    if formatter is None:
        fmt, scale, sz = _number_format(self,
                                        min_sz=5 if not print_full_mat else 0)
    else:
        fmt, scale, sz = formatter
    nColumnPerLine = int(math.floor((PRINT_OPTS.linewidth - len(indent)) / (sz + 1)))
    strt = ''
    firstColumn = 0

    if not force_truncate and \
       (self.size < PRINT_OPTS.threshold or print_full_mat):
        while firstColumn < self.shape[1]:
            lastColumn = min(firstColumn + nColumnPerLine - 1, self.shape[1] - 1)
            if nColumnPerLine < self.shape[1]:
                strt += '\n' if firstColumn != 1 else ''
                strt += 'Columns {} to {} \n{}'.format(
                    firstColumn, lastColumn, indent)
            if scale != 1:
                strt += SCALE_FORMAT.format(scale)
            for l in _range(self.shape[0]):
                strt += indent + (' ' if scale != 1 else '')
                row_slice = self[l, firstColumn:lastColumn + 1]
                strt += ' '.join(fmt.format(val / scale) for val in row_slice)
                strt += '\n'
            firstColumn = lastColumn + 1
    else:
        if scale != 1:
            strt += SCALE_FORMAT.format(scale)
        if has_vdots and has_hdots:
            vdotfmt = "{:^" + str((sz + 1) * n - 1) + "}"
            ddotfmt = u"{:^5}"
            for row in self[:n]:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
            strt += indent + ' '.join([vdotfmt.format('...'),
                                       ddotfmt.format(u'\u22F1'),
                                       vdotfmt.format('...')]) + "\n"
            for row in self[-n:]:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
        elif not has_vdots and has_hdots:
            for row in self:
                strt += __repr_row(row, indent, fmt, scale, sz, n)
        elif has_vdots and not has_hdots:
            vdotfmt = u"{:^" + \
                str(len(__repr_row(self[0], '', fmt, scale, sz))) + \
                "}\n"
            for row in self[:n]:
                strt += __repr_row(row, indent, fmt, scale, sz)
            strt += vdotfmt.format(u'\u22EE')
            for row in self[-n:]:
                strt += __repr_row(row, indent, fmt, scale, sz)
        else:
            for row in self:
                strt += __repr_row(row, indent, fmt, scale, sz)
    size_str = 'x'.join(str(size) for size in self.shape)
    strt += '[{} of size {}]\n'.format(type_str, size_str)
    return '\n' + strt


def _vector_str(self):
    type_str = self.__module__ + '.' + self.__class__.__name__
    self = self.numpy()
    if self.size == 0:
        return '[{} with no elements]\n'.format(type_str)
    fmt, scale, sz = _number_format(self)
    strt = ''
    ident = ''
    n = PRINT_OPTS.edgeitems
    dotfmt = u"{:^" + str(sz) + "}\n"
    if scale != 1:
        strt += SCALE_FORMAT.format(scale)
        ident = ' '
    if self.size < PRINT_OPTS.threshold:
        strt = (strt +
                '\n'.join(ident + fmt.format(val / scale) for val in self) +
                '\n')
    else:
        strt = (strt +
                '\n'.join(ident + fmt.format(val / scale) for val in self[:n]) +
                '\n' + (ident + dotfmt.format(u"\u22EE")) +
                '\n'.join(ident + fmt.format(val / scale) for val in self[-n:]) +
                '\n')
    strt += '[{} of size {}]\n'.format(type_str, self.size)
    return '\n' + strt
