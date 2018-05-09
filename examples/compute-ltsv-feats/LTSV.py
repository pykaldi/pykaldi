from __future__ import division
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb

def ApplySigmoidScale (ltsv_sigmoidThr, ltsv_sigmoidSlope, ltsv_input):
  ltsv_input = 1/( np.exp( (-1/ltsv_sigmoidSlope)*(ltsv_input-ltsv_sigmoidThr) )+1 )
  return ltsv_input



def ApplyLTSV(spec, ctx_win, sigThresh, sigSlope, sigmoidscale):

  nframes, nfeats = spec.shape
  if nframes < ctx_win+1:
    ctx_win = num_frames-1
  
  featsin = np.zeros((nframes+ctx_win, nfeats))
  featsin[0:nframes, :] = spec
  featsin[nframes:, :] = spec[nframes-ctx_win:,:]
  
  ltsv = np.zeros(nframes)
  ltsv_bins = np.zeros([ctx_win,nfeats])
  ltsv_bins_log = np.zeros([ctx_win,nfeats])
  entropy_vec = np.zeros(nfeats)

  ltsv_val=0
  
  for k in range(nframes):
    if k < int(round(ctx_win/2)):
      ltsv_bins[0:int(round(ctx_win/2))-k,:] = np.array([featsin[0,:],] * int(round(ctx_win/2)-k))
      ltsv_bins[int(round(ctx_win/2))-1-k:ctx_win,:] = featsin[0:int(round(ctx_win/2))+k+1,:]
    else:
      ltsv_bins = featsin[k-int(round(ctx_win/2))+1:k+int(round(ctx_win/2))+1,:];
  
    # this should never happen after ARMA
    if np.any(ltsv_bins[ltsv_bins<0]): 
      ltsv_bins[ltsv_bins<0] = 1/100 

    moving_context = np.sum(ltsv_bins,axis=0)
    ltsv_bins = (ltsv_bins / moving_context[None,:])

    # entropy
    ltsv_bins_log = np.log(ltsv_bins)
    ltsv_bins =  ltsv_bins*ltsv_bins_log*(-1) 
    entropy_vec = np.sum(ltsv_bins,axis=0)
  
    #variance
    entropy_vec = entropy_vec - (np.sum(entropy_vec)/nfeats)
    entropy_vec = np.power(entropy_vec, 2) 
 
    if k  < nframes - int(round(ctx_win/2)):
      ltsv_val = np.sum(entropy_vec)/nfeats
    ltsv[k] = ltsv_val
  
  if sigmoidscale:
    ltsv = ApplySigmoidScale(sigThresh, sigSlope, ltsv)

  return ltsv

