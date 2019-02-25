"""
PyKaldi defines the following CPU vector/matrix types:

================= ====================== ============================ =========================
Type              32-bit floating point  64-bit floating point        Other
================= ====================== ============================ =========================
Dense Matrix      :class:`.Matrix`       :class:`.DoubleMatrix`       :class:`.CompressedMatrix`
Dense Vector      :class:`.Vector`       :class:`.DoubleVector`
Symmetric Matrix  :class:`.SpMatrix`     :class:`.DoubleSpMatrix`
Triangular Matrix :class:`.TpMatrix`     :class:`.DoubleTpMatrix`
Sparse Matrix     :class:`.SparseMatrix` :class:`.DoubleSparseMatrix`
Sparse Vector     :class:`.SparseVector` :class:`.DoubleSparseMatrix`
================= ====================== ============================ =========================

In addition, there is a :class:`.GeneralMatrix` type which is a wrapper around
:class:`.Matrix`, :class:`.SparseMatrix` and :class:`.CompressedMatrix` types.

The dense :class:`Vector`/:class:`Matrix` types come in two flavors.

:class:`Vector`/:class:`Matrix` instances own the memory buffers backing them.
Instantiating a new :class:`Vector`/:class:`Matrix` object allocates new memory
for storing the elements. They support destructive operations that reallocate
memory.

:class:`SubVector`/:class:`SubMatrix` instances, on the other hand, share the
memory buffers owned by other objects. Instantiating a new
:class:`SubVector`/:class:`SubMatrix` object does not allocate new memory. Since
they provide views into other existing objects, they do not support destructive
operations that reallocate memory. Other than this caveat, they are equivalent
to :class:`Vector`/:class:`Matrix` instances for all practical purposes. Almost
any function or method accepting a :class:`Vector`/:class:`Matrix` instance can
instead be passed a :class:`SubVector`/:class:`SubMatrix` instance.

.. note::
    All mutating vector/matrix methods are marked with an underscore suffix.
    These methods overwrite the contents and return the resulting object,
    unless they have other return values, to support method chaining.
"""

from ._matrix import *
from ._str import set_printoptions


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
