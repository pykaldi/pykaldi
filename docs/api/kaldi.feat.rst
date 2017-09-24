kaldi\.feat
===========

.. automodule:: kaldi.feat

   
   
   

   
   
   

   
   
   
kaldi\.feat\.fbank
------------------

.. automodule:: kaldi.feat.fbank

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      Fbank
      FbankComputer
      FbankOptions
   
   

   
   
   
kaldi\.feat\.functions
----------------------

.. automodule:: kaldi.feat.functions

   
   
   .. rubric:: Functions

   .. autosummary::
   
      compute_deltas
      compute_power_spectrum
      compute_shift_deltas
      init_idft_bases
      reverse_frames
      sliding_window_cmn
      splice_frames
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      DeltaFeatures
      DeltaFeaturesOptions
      ShiftedDeltaFeatures
      ShiftedDeltaFeaturesOptions
      SlidingWindowCmnOptions
   
   

   
   
   
kaldi\.feat\.mel
----------------

.. automodule:: kaldi.feat.mel

   
   
   .. rubric:: Functions

   .. autosummary::
   
      compute_lifter_coeffs
      compute_lpc
      get_equal_loudness_vector
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      MelBanks
      MelBanksOptions
   
   

   
   
   
kaldi\.feat\.mfcc
-----------------

.. automodule:: kaldi.feat.mfcc

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      Mfcc
      MfccComputer
      MfccOptions
   
   

   
   
   
kaldi\.feat\.online
-------------------

.. automodule:: kaldi.feat.online

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      OnlineAppendFeature
      OnlineCacheFeature
      OnlineCmvn
      OnlineCmvnOptions
      OnlineCmvnState
      OnlineDeltaFeature
      OnlineFbank
      OnlineMatrixFeature
      OnlineMfcc
      OnlinePlp
      OnlineSpliceFrames
      OnlineSpliceOptions
      OnlineTransform
   
   

   
   
   
kaldi\.feat\.pitch
------------------

.. automodule:: kaldi.feat.pitch

   
   
   .. rubric:: Functions

   .. autosummary::
   
      compute_and_process_kaldi_pitch
      compute_kaldi_pitch
      process_pitch
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      OnlinePitchFeature
      OnlineProcessPitch
      PitchExtractionOptions
      ProcessPitchOptions
   
   

   
   
   
kaldi\.feat\.plp
----------------

.. automodule:: kaldi.feat.plp

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      Plp
      PlpComputer
      PlpOptions
   
   

   
   
   
kaldi\.feat\.signal
-------------------

.. automodule:: kaldi.feat.signal

   
   
   .. rubric:: Functions

   .. autosummary::
   
      convolve_signals
      downsample_wave_form
      fft_based_block_convolve_signals
      fft_based_convolve_signals
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      ArbitraryResample
      LinearResample
   
   

   
   
   
kaldi\.feat\.spectrogram
------------------------

.. automodule:: kaldi.feat.spectrogram

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      Spectrogram
      SpectrogramComputer
      SpectrogramOptions
   
   

   
   
   
kaldi\.feat\.wave
-----------------

.. automodule:: kaldi.feat.wave

   
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      WaveData
   
   

   
   
   
kaldi\.feat\.window
-------------------

.. automodule:: kaldi.feat.window

   
   
   .. rubric:: Functions

   .. autosummary::
   
      dither
      extract_waveform_remainder
      first_sample_of_frame
      num_frames
      preemphasize
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
   
      FeatureWindowFunction
      FrameExtractionOptions
   
   

   
   
   