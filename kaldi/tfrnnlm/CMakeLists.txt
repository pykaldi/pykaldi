set(PACKAGE "${PACKAGE}.tfrnnlm")

if(TFRNNLM)

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TFRNNLM_CXX_FLAGS}")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${TF_LIB_DIR}")
  link_directories(${TF_LIB_DIR})

  add_pyclif_library("_tensorflow_rnnlm" tensorflow-rnnlm.clif
    CLIF_DEPS _options_itf _arc _deterministic_fst
    LIBRARIES kaldi-tensorflow-rnnlm tensorflow_cc tensorflow_framework
  )

endif(TFRNNLM)
