kaldi\.matrix
=============

.. automodule:: kaldi.matrix

   
   
   .. rubric:: Functions

   .. autosummary::
   
      construct_matrix
      construct_vector
      set_printoptions
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      Matrix
      MatrixResizeType
      MatrixStrideType
      MatrixTransposeType
      SpCopyType
      SubMatrix
      SubVector
      Vector
   
   

   
   
   
kaldi\.matrix\.compressed
-------------------------

.. automodule:: kaldi.matrix.compressed

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      CompressedMatrix
      CompressionMethod
   
   

   
   
   
kaldi\.matrix\.functions
------------------------

.. automodule:: kaldi.matrix.functions

   
   
   .. rubric:: Functions

   .. autosummary::
   
      assert_same_dim_matrix
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      MatrixExponential
   
   

   
   
   
kaldi\.matrix\.matrix
---------------------

.. automodule:: kaldi.matrix.matrix

   
   
   .. rubric:: Functions

   .. autosummary::
   
      approx_equal_double_matrix
      approx_equal_matrix
      assert_equal_double_matrix
      assert_equal_matrix
      construct_matrix
      create_eigenvalue_double_matrix
      create_eigenvalue_matrix
      matrix_to_numpy
      read_htk
      same_dim_double_matrix
      same_dim_matrix
      sort_double_svd
      sort_svd
      trace_double_mat
      trace_double_mat_mat
      trace_double_mat_mat_mat
      trace_double_mat_mat_mat_mat
      trace_mat
      trace_mat_mat
      trace_mat_mat_mat
      trace_mat_mat_mat_mat
      wrie_htk
      wrie_sphinx
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      DoubleMatrix
      DoubleMatrixBase
      HtkHeader
      Matrix
      MatrixBase
      MatrixResizeType
      MatrixStrideType
      MatrixTransposeType
      SubMatrix
   
   

   
   
   
kaldi\.matrix\.optimization
---------------------------

.. automodule:: kaldi.matrix.optimization

   
   
   .. rubric:: Functions

   .. autosummary::
   
      linear_cgd
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      LbfgsOptions
      LinearCgdOptions
      OptimizeLbfgs
   
   

   
   
   
kaldi\.matrix\.packed
---------------------

.. automodule:: kaldi.matrix.packed

   
   
   .. rubric:: Functions

   .. autosummary::
   
      approx_equal_sp_matrix
      assert_equal_sp_matrix
      solve_double_quadratic_matrix_problem
      solve_quadratic_matrix_problem
      solve_quadratic_problem
      trace_double_sp_sp
      trace_mat_sp_mat
      trace_mat_sp_mat_sp
      trace_sp_mat
      trace_sp_sp
      trace_sp_sp_lower
      vec_sp_vec
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      DoubleSpMatrix
      Matrix
      MatrixResizeType
      PackedMatrix
      SolverOptions
      SpMatrix
      TpMatrix
   
   

   
   
   
kaldi\.matrix\.sparse
---------------------

.. automodule:: kaldi.matrix.sparse

   
   
   .. rubric:: Functions

   .. autosummary::
   
      extract_row_range_with_padding
      filter_compressed_matrix_rows
      filter_general_matrix_rows
      filter_matrix_rows
      filter_sparse_matrix_rows
      trace_mat_smat
      vec_svec
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      GeneralMatrix
      GeneralMatrixType
      SparseMatrix
      SparseVector
   
   

   
   
   
kaldi\.matrix\.vector
---------------------

.. automodule:: kaldi.matrix.vector

   
   
   .. rubric:: Functions

   .. autosummary::
   
      ApproxEqualVector
      approx_equal
      assert_equal_double_vector
      assert_equal_vector
      construct_vector
      double_vec_vec
      vec_mat_vec
      vec_vec
      vector_to_numpy
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      DoubleVector
      DoubleVectorBase
      MatrixResizeType
      MatrixTransposeType
      SubVector
      Vector
      VectorBase
   
   

   
   
   