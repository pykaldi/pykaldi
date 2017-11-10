
from . import _compressed_matrix
from . import _kaldi_matrix
from . import _kaldi_matrix_ext
from . import _kaldi_vector
from . import _kaldi_vector_ext
import _matrix_common # FIXME: Relative/absolute import is buggy in Python 3.
from . import _sparse_matrix
from . import _sp_matrix
from . import _tp_matrix

from ._matrix_functions import *
from ._sp_matrix import SolverOptions
from ._sp_matrix import solve_quadratic_problem
from ._sp_matrix import solve_quadratic_matrix_problem
from ._sp_matrix import solve_double_quadratic_matrix_problem


def approx_equal(a, b, tol=0.01):
    """Checks if given vectors (or matrices) are approximately equal.

    Args:
        a (Vector or Matrix or SpMatrix or DoubleVector or DoubleMatrix or DoubleSpMatrix):
            The first object.
        b (Vector or Matrix or SpMatrix or DoubleVector or DoubleMatrix or DoubleSpMatrix):
            The second object.
        tol (float): The tolerance for the equality check. Defaults to ``0.01``.

    Returns:
        True if input objects have the same type and size, and
        :math:`\\Vert a-b \\Vert \\leq \\mathrm{tol} \\times \\Vert a \\Vert`.

    Raises:
        TypeError: If the first object is not a vector or matrix instance.
    """
    if isinstance(a, (_kaldi_vector.VectorBase, _kaldi_vector.DoubleVectorBase,
                      _kaldi_matrix.MatrixBase, _kaldi_matrix.DoubleMatrixBase,
                      _sp_matrix.SpMatrix, _sp_matrix.DoubleSpMatrix)):
        return a.approx_equal(b, tol)
    raise TypeError("a is not a vector or matrix instance")


def assert_equal(a, b, tol=0.01):
    """Asserts given vectors (or matrices) are approximately equal.

    Args:
        a (Vector or Matrix or SpMatrix or DoubleVector or DoubleMatrix or DoubleSpMatrix):
            The first object.
        b (Vector or Matrix or SpMatrix or DoubleVector or DoubleMatrix or DoubleSpMatrix):
            The second object.
        tol (float): The tolerance for the equality check. Defaults to ``0.01``.

    Raises:
        TypeError: If the first object is not a vector or matrix instance.
        AssertionError: If input objects do not have the same type or size, or
        :math:`\\Vert a-b \\Vert > \\mathrm{tol} \\times \\Vert a \\Vert`.
    """
    assert(approx_equal(a, b, tol))


def create_eigenvalue_matrix(real, imag, D=None):
    """Creates the eigenvalue matrix.

    Eigenvalue matrix :math:`D` is part of the decomposition used in eig.
    :math:`D` will be block-diagonal with blocks of size 1 (for real
    eigenvalues) or 2x2 for complex pairs. If a complex pair is :math:`\\lambda
    +- i\\mu`, :math:`D` will have a corresponding 2x2 block :math:`[\\lambda,
    \\mu; -\\mu, \\lambda]`. This function will throw if any complex eigenvalues
    are not in complex conjugate pairs (or the members of such pairs are not
    consecutively numbered). The D you supply must has correct dimensions.

    Args:
        real (Vector or DoubleVector): The real part of the eigenvalues.
        imag (Vector or DoubleVector): The imaginary part of the eigenvalues.
        D (Matrix or DoubleMatrix or None): The output matrix.
            If provided, the eigenvalue matrix is written into this matrix.
            If ``None``, the eigenvalue matrix is returned.
            Defaults to ``None``.

    Returns:
        Matrix or DoubleMatrix: The eigenvalue matrix if **D** is ``None``.

    Raises:
        RuntimeError: If `real.dim != imag.dim`
        TypeError: If input types are not supported.
    """
    if (isinstance(real, _kaldi_vector.VectorBase) and
        isinstance(imag, _kaldi_vector.VectorBase)):
        if D is None:
            D = Matrix(real.dim, real.dim)
            _kaldi_matrix._create_eigenvalue_matrix(real, imag, D)
            return D
        else:
            _kaldi_matrix._create_eigenvalue_matrix(real, imag, D)
    if (isinstance(real, _kaldi_vector.DoubleVectorBase) and
        isinstance(imag, _kaldi_vector.DoubleVectorBase)):
        if D is None:
            D = DoubleMatrix(real.dim, real.dim)
            _kaldi_matrix._create_eigenvalue_double_matrix(real, imag, D)
            return D
        else:
            _kaldi_matrix._create_eigenvalue_double_matrix(real, imag, D)
    raise TypeError("real and imag should be vectors with the same data type.")


def sort_svd(s, U, Vt=None, sort_on_absolute_value=True):
    """Sorts singular-value decomposition in-place.

    SVD is :math:`U\\ diag(s)\\ V^T`.

    This function is as generic as possible, to be applicable to other
    types of problems. Requires `s.dim == U.num_cols`, and sorts from
    greatest to least absolute value, moving the columns of **U**,
    and the rows of **Vt**, if provided, around in the same way.

    Note:
        The ``absolute value'' part won't matter if this is an actual SVD,
        since singular values are non-negative.

    Args:
        s (Vector): The singular values.
        U (Matrix): The :math:`U` part of SVD.
        Vt (Matrix): The :math:`V^T` part of SVD. Defaults to ``None``.
        sort_on_absolute_value (bool): How to sort **s**.
            If True, sort from greatest to least absolute value. Otherwise,
            sort from greatest to least value. Defaults to ``True``.

    Raises:
        RuntimeError: If `s.dim != U.num_cols`.
        TypeError: If input types are not supported.
    """
    if (isinstance(s, _kaldi_vector.VectorBase) and
        isinstance(U, _kaldi_matrix.MatrixBase)):
        _kaldi_matrix._sort_svd(s, U, Vt, sort_on_absolute_value)
    if (isinstance(s, _kaldi_vector.DoubleVectorBase) and
        isinstance(U, _kaldi_matrix.DoubleMatrixBase)):
        _kaldi_matrix._sort_double_svd(s, U, Vt, sort_on_absolute_value)
    raise TypeError("s and U should respectively be a vector and matrix with "
                    "matching data types.")


def filter_matrix_rows(matrix, keep_rows):
    """Filters matrix rows.

    The output is a matrix containing only the rows `r` of **in** such that
    `keep_rows[r] == True`.

    Args:
        matrix (Matrix or SparseMatrix or CompressedMatrix or GeneralMatrix or DoubleMatrix or DoubleSparseMatrix):
            The input matrix.
        keep_rows (List[bool]): The list that determines which rows to keep.

    Returns:
        A new matrix constructed with the rows to keep.

    Raises:
        RuntimeError: If `matrix.num_rows != keep_rows.length`.
        TypeError: If input matrix type is not supported.
    """
    if isinstance(matrix, _kaldi_matrix.Matrix):
        return _sparse_matrix._filter_matrix_rows(matrix, keep_rows)
    if isinstance(matrix, _sparse_matrix.SparseMatrix):
        return _sparse_matrix._filter_sparse_matrix_rows(matrix, keep_rows)
    if isinstance(matrix, _compressed_matrix.CompressedMatrix):
        return _sparse_matrix._filter_compressed_matrix_rows(matrix, keep_rows)
    if isinstance(matrix, _sparse_matrix.GeneralMatrix):
        return _sparse_matrix._filter_general_matrix_rows(matrix, keep_rows)
    if isinstance(matrix, _kaldi_matrix.DoubleMatrix):
        return _sparse_matrix._filter_matrix_rows_double(matrix, keep_rows)
    if isinstance(matrix, _sparse_matrix.DoubleSparseMatrix):
        return _sparse_matrix._filter_sparse_matrix_rows_double(matrix, keep_rows)

    raise TypeError("input matrix type is not supported.")


def vec_vec(v1, v2):
    """Returns the dot product of vectors.

    Args:
        v1 (Vector or DoubleVector): The first vector.
        v2 (Vector or DoubleVector or SparseVector or DoubleSparseVector):
            The second vector.

    Returns:
        The dot product of v1 and v2.

    Raises:
        RuntimeError: In case of size mismatch.
        TypeError: If input types are not supported.
    """
    if isinstance(v1, _kaldi_vector.VectorBase):
        if isinstance(v2, _kaldi_vector.VectorBase):
            return _kaldi_vector._vec_vec(v1, v2)
        elif isinstance(v2, _sparse_matrix.SparseVector):
            return _sparse_matrix._vec_svec(v1, v2)
    elif isinstance(v1, _kaldi_vector.DoubleVectorBase):
        if isinstance(v2, _kaldi_vector.DoubleVectorBase):
            return _kaldi_vector._vec_vec_double(v1, v2)
        elif isinstance(v2, _sparse_matrix.DoubleSparseVector):
            return _sparse_matrix._vec_svec_double(v1, v2)

    raise TypeError("v1 and v2 should be vectors with the same data type.")


def vec_mat_vec(v1, M, v2):
    """Computes a vector-matrix-vector product.

    Performs the operation :math:`v_1\\ M\\ v_2`.

    Precision of input matrices should match.

    Args:
        v1 (Vector or DoubleVector): The first input vector.
        M (Matrix or DoubleMatrix or SpMatrix): The input matrix.
        v2 (Vector or DoubleVector): The second input vector.

    Returns:
       The vector-matrix-vector product.

    Raises:
       RuntimeError: In case of size mismatch.
    """
    if (isinstance(v1, _kaldi_vector.VectorBase) and
        isinstance(v2, _kaldi_vector.VectorBase)):
        if isinstance(M, _kaldi_matrix.MatrixBase):
            return _kaldi_vector_ext._vec_mat_vec(v1, M, v2)
        if isinstance(M, _sp_matrix.SpMatrix):
            return _sp_matrix._vec_sp_vec(v1, M, v2)
    elif (isinstance(v1, _kaldi_vector.DoubleVectorBase) and
          isinstance(v2, _kaldi_vector.DoubleVectorBase)):
        if isinstance(M, _kaldi_matrix.DoubleMatrixBase):
            return _kaldi_vector_ext._vec_mat_vec_double(v1, M, v2)
        if isinstance(M, _sp_matrix.DoubleSpMatrix):
            return _sp_matrix._vec_sp_vec_double(v1, M, v2)

    raise TypeError("given combination of input types is not supported")

def trace_mat(A):
    """Returns the trace of :math:`A`.

    Args:
        A (Matrix or DoubleMatrix): The input matrix.
    """
    if isinstance(A, _kaldi_matrix.MatrixBase):
        return _kaldi_matrix._trace_mat(A)
    if isinstance(A, _kaldi_matrix.DoubleMatrixBase):
        return _kaldi_matrix._trace_double_mat(A)
    raise TypeError("input matrix type is not supported")


def trace_mat_mat(A, B, transA=_matrix_common.MatrixTransposeType.NO_TRANS):
    """Returns the trace of :math:`A\\ B`.

    Precision of input matrices should match.

    Args:
        A (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix or SparseMatrix or DoubleSparseMatrix):
            The first input matrix.
        B (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix or SparseMatrix or DoubleSparseMatrix):
            The second input matrix.
        transA (_matrix_common.MatrixTransposeType):
            Whether to use **A** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        lower (bool): Whether to count lower-triangular elements only once.
            Active only if both inputs are symmetric matrices.
            Defaults to ``False``.
    """
    if isinstance(A, _kaldi_matrix.MatrixBase):
        if isinstance(B, _kaldi_matrix.MatrixBase):
            return _kaldi_matrix._trace_mat_mat(A, B, transA)
        elif isinstance(B, _sparse_matrix.SparseMatrix):
            return _sparse_matrix._trace_mat_smat(A, B, transA)
    elif isinstance(A, _sp_matrix.SpMatrix):
        if isinstance(B, _kaldi_matrix.MatrixBase):
            return _sp_matrix._trace_sp_mat(A, B)
        elif isinstance(B, _sp_matrix.SpMatrix):
            if lower:
                return _sp_matrix._trace_sp_sp_lower(A, B)
            else:
                return _sp_matrix._trace_sp_sp(A, B)
    elif isinstance(A, _kaldi_matrix.DoubleMatrixBase):
        if isinstance(B, _kaldi_matrix.DoubleMatrixBase):
            return _kaldi_matrix._trace_double_mat_mat(A, B, transA)
        elif isinstance(B, _sparse_matrix.DoubleSparseMatrix):
            return _sparse_matrix._trace_double_mat_smat(A, B, transA)
    elif isinstance(A, _sp_matrix.DoubleSpMatrix):
        if isinstance(B, _sp_matrix.DoubleSpMatrix):
            if lower:
                return _sp_matrix._trace_double_sp_sp_lower(A, B)
            else:
                return _sp_matrix._trace_double_sp_sp(A, B)

    raise TypeError("given combination of matrix types is not supported")


def trace_mat_mat_mat(A, B, C,
                      transA=_matrix_common.MatrixTransposeType.NO_TRANS,
                      transB=_matrix_common.MatrixTransposeType.NO_TRANS,
                      transC=_matrix_common.MatrixTransposeType.NO_TRANS):
    """Returns the trace of :math:`A\\ B\\ C`.

    Precision of input matrices should match.

    Args:
        A (Matrix or DoubleMatrix): The first input matrix.
        B (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix):
            The second input matrix.
        C (Matrix or DoubleMatrix): The third input matrix.
        transA (_matrix_common.MatrixTransposeType):
            Whether to use **A** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        transB (_matrix_common.MatrixTransposeType):
            Whether to use **B** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        transC (_matrix_common.MatrixTransposeType):
            Whether to use **C** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
    """
    if isinstance(A, _kaldi_matrix.MatrixBase):
        if (isinstance(B, _kaldi_matrix.MatrixBase) and
            isinstance(C, _kaldi_matrix.MatrixBase)):
            return _kaldi_matrix._trace_mat_mat_mat(A, transA, B, transB,
                                                    C, transC)
        elif (isinstance(B, _sp_matrix.SpMatrix) and
              isinstance(C, _kaldi_matrix.MatrixBase)):
            return _sp_matrix._trace_mat_sp_mat(A, transA, B, C, transC)
    elif isinstance(A, _kaldi_matrix.DoubleMatrixBase):
        if (isinstance(B, _kaldi_matrix.DoubleMatrixBase) and
            isinstance(C, _kaldi_matrix.DoubleMatrixBase)):
            return _kaldi_matrix._trace_double_mat_mat_mat(A, transA, B, transB,
                                                           C, transC)
        elif (isinstance(B, _sp_matrix.DoubleSpMatrix) and
              isinstance(C, _kaldi_matrix.DoubleMatrixBase)):
            return _sp_matrix._trace_double_mat_sp_mat(A, transA, B, C, transC)

    raise TypeError("given combination of matrix types is not supported")


def trace_mat_mat_mat_mat(A, B, C, D,
                          transA=_matrix_common.MatrixTransposeType.NO_TRANS,
                          transB=_matrix_common.MatrixTransposeType.NO_TRANS,
                          transC=_matrix_common.MatrixTransposeType.NO_TRANS,
                          transD=_matrix_common.MatrixTransposeType.NO_TRANS):
    """Returns the trace of :math:`A\\ B\\ C\\ D`.

    Precision of input matrices should match.

    Args:
        A (Matrix or DoubleMatrix): The first input matrix.
        B (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix):
            The second input matrix.
        C (Matrix or DoubleMatrix): The third input matrix.
        D (Matrix or DoubleMatrix or SpMatrix or DoubleSpMatrix):
            The fourth input matrix.
        transA (_matrix_common.MatrixTransposeType):
            Whether to use **A** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        transB (_matrix_common.MatrixTransposeType):
            Whether to use **B** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        transC (_matrix_common.MatrixTransposeType):
            Whether to use **C** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
        transD (_matrix_common.MatrixTransposeType):
            Whether to use **D** or its transpose.
            Defaults to ``MatrixTransposeType.NO_TRANS``.
    """
    if isinstance(A, _kaldi_matrix.MatrixBase):
        if (isinstance(B, _kaldi_matrix.MatrixBase) and
            isinstance(C, _kaldi_matrix.MatrixBase) and
            isinstance(D, _kaldi_matrix.MatrixBase)):
            return _kaldi_matrix._trace_mat_mat_mat_mat(A, transA, B, transB,
                                                        C, transC, D, transD)
        elif (isinstance(B, _sp_matrix.SpMatrix) and
              isinstance(C, _kaldi_matrix.MatrixBase) and
              isinstance(D, _sp_matrix.SpMatrix)):
            return _sp_matrix._trace_mat_sp_mat_sp(A, transA, B, C, transC, D)
    elif isinstance(A, _kaldi_matrix.DoubleMatrixBase):
        if (isinstance(B, _kaldi_matrix.DoubleMatrixBase) and
            isinstance(C, _kaldi_matrix.DoubleMatrixBase) and
            isinstance(D, _kaldi_matrix.DoubleMatrixBase)):
            return _kaldi_matrix._trace_double_mat_mat_mat_mat(
                A, transA, B, transB, C, transC, D, transD)
        elif (isinstance(B, _sp_matrix.DoubleSpMatrix) and
              isinstance(C, _kaldi_matrix.DoubleMatrixBase) and
              isinstance(D, _sp_matrix.DoubleSpMatrix)):
            return _sp_matrix._trace_double_mat_sp_mat_sp(A, transA, B,
                                                          C, transC, D)

    raise TypeError("given combination of matrix types is not supported")

################################################################################

__all__ = [name for name in dir() if name[0] != '_']
