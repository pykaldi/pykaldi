from __future__ import division
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb
from kaldi.matrix.functions import compute_dct_matrix
from kaldi.matrix import Matrix


def DCTFCompute(input_s, input_dct_matrix, ctx_win, num_ceps):
  """This function computes the Discrete Cosine Transform of a signal.

  Args:
      input_s: the input signal
      input_dct_matrix: The DCT matrix
      ctx_win: the context window
      num_ceps: the number of coefficients
          
  Returns:
      dctf_output: The transformed singal
  """
  input_row_n = input_s.shape[0]
  fr_len = ctx_win
  fr_sh = 1;

  num_samples = input_s.shape[0]
  n_fr = 1 + np.floor( (num_samples-fr_len)/fr_sh ).astype('int')
  
  transformed_input = np.zeros((n_fr, num_ceps))

  for r_ind in range(n_fr):
    start = r_ind*fr_sh
    end = start+fr_len  
    window = input_s[start:end]
    transformed_input[r_ind,:] = input_dct_matrix.dot(window)
  
  dctf_output = np.zeros((input_row_n, num_ceps))
  dctf_output[0:n_fr,:] = transformed_input
  
  for ctx_index in range(ctx_win-1):
    dctf_output[n_fr+ctx_index,:] = transformed_input[-1,:]
  return dctf_output


def ApplyDCT(num_cep, context_window, feature):
  """This function applies the Discrete Cosine Transform to a feature.

  Args:
      num_cep: the number of DCT coefficients.
      context_window: window over which we will calculate the DCT. 
      feature: the input feature
          
  Returns:
      ltsv: The LTSV features
  """
  dct_matrix_full = Matrix(context_window, context_window)
  compute_dct_matrix(dct_matrix_full)
  dct_matrix_full = dct_matrix_full.numpy()
  dct_matrix = dct_matrix_full[0:num_cep,:]
 
  final_out = DCTFCompute(feature, dct_matrix, context_window, num_cep)
  final_out = final_out[:,0];
 
  return final_out

