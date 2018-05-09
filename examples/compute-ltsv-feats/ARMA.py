from __future__ import division
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb


def ApplyARMA(in_matrix, order):

  nframes, nfeats = in_matrix.shape
  out_matrix = np.copy(in_matrix)

  for c_ind in range(nfeats):
    tmp1 = 0
    tmp2 = 0
    for r_ind in range(nframes - order):
      if r_ind < order:
        out_matrix[r_ind, c_ind] = 0.01 * in_matrix[r_ind, c_ind]
      elif r_ind == order:
        for cnt in range(order):
          tmp1 = tmp1 + out_matrix[r_ind-1-cnt, c_ind]
          tmp2 = tmp2 + in_matrix[r_ind+cnt, c_ind]
        tmp2 = tmp2 + in_matrix[r_ind+order ,c_ind]
        out_matrix[r_ind, c_ind] = (tmp1+tmp2)/(2*order + 1)
      else:
        tmp1 = tmp1 + out_matrix[r_ind-1, c_ind] - out_matrix[r_ind-1-order, c_ind]
        tmp2 = tmp2 + in_matrix[r_ind+order, c_ind] - in_matrix[r_ind-1, c_ind]
        out_matrix[r_ind, c_ind] = (tmp1+tmp2)/(2*order+1)
  
  return out_matrix
