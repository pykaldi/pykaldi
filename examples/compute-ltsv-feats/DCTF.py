from __future__ import division
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb

def DctMatrix(num_of_rows, num_of_cols):

  dct_matrix_out = np.zeros((num_of_rows, num_of_cols))
  normalizer = np.sqrt(1/num_of_cols)
  dct_matrix_out[0,:]=normalizer

  normalizer = np.sqrt(2/num_of_cols)

  for row in range(1, num_of_rows):
    for col in range(num_of_cols):
      dct_matrix_out[row,col] = normalizer * np.cos( (np.pi/num_of_cols)*(col+0.5)*row ) 

  return dct_matrix_out


def DCTFCompute(input_s, input_dct_matrix, ctx_win, num_ceps):
  input_row_n = input_s.shape[0]
  fr_len = ctx_win
  fr_sh = 1;

  num_samples = input_s.shape[0]
  n_fr = 1 + np.floor( (num_samples-fr_len)/fr_sh ).astype('int')
  
  dctf_output = np.zeros((n_fr, num_ceps))

  for r_ind in range(n_fr):
    start = r_ind*fr_sh
    end = start+fr_len  
    window = input_s[start:end]
    dctf_output[r_ind,:] = input_dct_matrix.dot(window)
  
  output = np.zeros((input_row_n, num_ceps))
  output[0:n_fr,:] = dctf_output
  
  for ctx_index in range(ctx_win-1):
    output[n_fr+ctx_index,:] = dctf_output[-1,:]
  return output


def ApplyDCT(num_cep, context_window, ltsv):

  dct_matrix_temp = DctMatrix(context_window, context_window)
  dct_matrix = np.zeros((num_cep,context_window))
  dct_matrix[:,:] = dct_matrix_temp[0:num_cep,:]
 
  final_out = DCTFCompute(ltsv, dct_matrix, context_window, num_cep)
  final_out = final_out[:,0];
 
  return final_out

