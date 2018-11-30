from __future__ import division
import numpy as np

def ApplyARMA(in_matrix, order):
  """This function applies Autoregressive Moving Averarge(ARMA)
     on the rows of the input matrix.
     
  Args:
      in_matrix: the input matrix
      order: the order of the ARMA process

  Returns:
      out_matrix: the matrix after the ARMA process
  """

  nframes, nfeats = in_matrix.shape
  out_matrix = np.copy(in_matrix)

  for c in range(nfeats):
    tmp1 = 0
    tmp2 = 0
    for r in range(nframes - order):
      if r < order:
        out_matrix[r, c] = 0.01 * in_matrix[r, c]
      elif r == order:
        for cnt in range(order):
          tmp1 = tmp1 + out_matrix[r-1-cnt, c]
          tmp2 = tmp2 + in_matrix[r+cnt, c]
        tmp2 = tmp2 + in_matrix[r+order ,c]
        out_matrix[r, c] = (tmp1+tmp2)/(2*order + 1)
      else:
        tmp1 = tmp1 + out_matrix[r-1, c] - out_matrix[r-1-order, c]
        tmp2 = tmp2 + in_matrix[r+order, c] - in_matrix[r-1, c]
        out_matrix[r, c] = (tmp1+tmp2)/(2*order+1)
  
  return out_matrix
