cmake_minimum_required(VERSION 3.5)

# Prevent in-source builds
if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

set(DOUBLE_PRECISION 0)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

####################################################################################################
if(NOT PYCLIF)
    set(PYCLIF $ENV{PYCLIF})
    if (NOT PYCLIF)
      find_program(PYCLIF pyclif)
      if(NOT PYCLIF)
          message(FATAL_ERROR "The 'pyclif' program was not found. Specify PYCLIF environmental variable or with option -DPYCLIF.")
      endif(NOT PYCLIF)
    endif(NOT PYCLIF)
endif(NOT PYCLIF)
message("Using PYCLIF: $PYCLIF")

if(NOT CLIF_INSTALL_DIR)
  set(CLIF_INSTALL_DIR $ENV{CLIF_INSTALL_DIR})
  if(NOT CLIF_INSTALL_DIR)
    string(REGEX REPLACE "/bin/pyclif$" "" CLIF_INSTALL_DIR ${PYCLIF})
    message("Environmental variable CLIF_INSTALL_DIR or option -DCLIF_INSTALL_DIR was not set. Defaulting to ${CLIF_INSTALL_DIR}.")
  endif(NOT CLIF_INSTALL_DIR)
endif(NOT CLIF_INSTALL_DIR)
message("Using CLIF_INSTALL_DIR: $CLIF_INSTALL_DIR")

# Find clif-matcher 
find_program(CLIF_MATCHER clif-matcher PATHS CLIF_INSTALL_DIR, CLIF_INSTALL_DIR/clang/bin)
if(NOT CLIF_MATCHER)
  message(FATAL_ERROR "The 'clif-matcher' program was not found in ${CLIF_INSTALL_DIR}.")
endif(NOT CLIF_MATCHER)

if(NOT KALDI_ROOT)
    set(KALDI_ROOT $ENV{KALDI_ROOT})
    if(NOT KALDI_ROOT)
      find_program(KALDI_ROOT kaldi)
      if(NOT KALDI_ROOT)
        message(FATAL_ERROR "You need to specify Kaldi path. Specify KALDI_ROOT environmental variable or with option -DKALDI_ROOT.")
      endif(NOT KALDI_ROOT)
    endif(NOT KALDI_ROOT)
endif(NOT KALDI_ROOT)

set(kaldisrc "${KALDI_ROOT}/src")
set(kalditools "${KALDI_ROOT}/tools")
set(ATLAS_ROOT_inc ${kalditools}/ATLAS/include)
set(OPENFST_ROOT_inc ${kalditools}/openfst/include)


####################################################################################################
# Function to set up rules to invoke pyclif on a .clif file and generate the wrapper .cc and .h files
####################################################################################################
function(add_pyclif_library name pyclif_file)
    cmake_parse_arguments(PYCLIF_LIBRARY "" "" "CC_DEPS;CLIF_DEPS;CXX_FLAGS" ${ARGN})

    string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
    set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}_clifwrap.cc")
    set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}_clifwrap.h")
    set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}_clifwrap_init.cc")

    string(REPLACE "-" "_" module_name ${pyclif_file_basename})

    add_custom_target(
        ${name} ALL
        BYPRODUCTS ${gen_cc} ${gen_h} ${gen_init}
        COMMAND
            ${PYCLIF}
            --matcher_bin=${CLIF_MATCHER}
            --ccdeps_out ${gen_cc} --header_out ${gen_h} --ccinit_out ${gen_init}
            --modname=${module_name}
            -I${CMAKE_BINARY_DIR}/kaldi
            "-f-I${PYTHON_INCLUDE_DIRS} \
               -I${CLIF_INSTALL_DIR}/.. \
               -I${kaldisrc} \
               -I.. \
               -I${OPENFST_ROOT_inc} \
               -DKALDI_DOUBLEPRECISION=${DOUBLE_PRECISION} \
               -DHAVE_EXECINFO_H=1 \
               -DHAVE_CXXABI_H \
               -DHAVE_ATLAS \
               -I${ATLAS_ROOT_inc} \
               -msse -msse2 -pthread -g -O0 \
               -DKALDI_PARANOID \
               -fPIC \
               -std=c++11 \
                 ${CMAKE_CXX_FLAGS}"
            ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
        VERBATIM
        DEPENDS ${PYCLIF_LIBRARY_CC_DEPS} ${PYCLIF_LIBRARY_CLIF_DEPS}
    )
endfunction(add_pyclif_library)


include_directories(${CLIF_INSTALL_DIR}/..)
add_subdirectory("kaldi")
