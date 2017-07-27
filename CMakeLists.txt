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
      message(FATAL_ERROR "Could not find 'pyclif'. Set PYCLIF environment variable or -DPYCLIF option.")
    endif(NOT PYCLIF)
  endif(NOT PYCLIF)
endif(NOT PYCLIF)
message("Using PYCLIF: ${PYCLIF}")

if(NOT CLIF_DIR)
  set(CLIF_DIR $ENV{CLIF_DIR})
  if(NOT CLIF_DIR)
    string(REGEX REPLACE "/bin/pyclif$" "" CLIF_DIR ${PYCLIF})
    message("Environment variable CLIF_DIR or option -DCLIF_DIR was not set. Defaulting to ${CLIF_DIR}.")
  endif(NOT CLIF_DIR)
endif(NOT CLIF_DIR)
message("Using CLIF_DIR: ${CLIF_DIR}")

# Find clif-matcher
find_program(CLIF_MATCHER clif-matcher PATHS ${CLIF_DIR}, ${CLIF_DIR}/clang/bin)
if(NOT CLIF_MATCHER)
  message(FATAL_ERROR "Could not find 'clif-matcher' in ${CLIF_DIR}/clang/bin/.")
endif(NOT CLIF_MATCHER)

if(NOT KALDI_DIR)
  set(KALDI_DIR $ENV{KALDI_DIR})
  if(NOT KALDI_DIR)
    message(FATAL_ERROR "You need to specify Kaldi directory. Set KALDI_DIR environmen variable or -DKALDI_DIR option.")
  endif(NOT KALDI_DIR)
endif(NOT KALDI_DIR)

set(KALDI_SRC_DIR "${KALDI_DIR}/src")
set(KALDI_TOOLS_DIR "${KALDI_DIR}/tools")
set(ATLAS_INC_DIR ${KALDI_TOOLS_DIR}/ATLAS/include)
set(OPENFST_INC_DIR ${KALDI_TOOLS_DIR}/openfst/include)


####################################################################################################
# Function to set up rules to invoke pyclif on a .clif file and generate the wrapper .cc and .h files
####################################################################################################
function(add_pyclif_library name pyclif_file)
    cmake_parse_arguments(PYCLIF_LIBRARY "" "" "CC_DEPS;CLIF_DEPS;CXX_FLAGS;NAMESPACES" ${ARGN})

    string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
    set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.cc")
    set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.h")
    set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap-init.cc")

    string(REPLACE "-" "_" module_name ${pyclif_file_basename})

    add_custom_command(
        OUTPUT ${gen_cc} ${gen_h} ${gen_init}
        COMMAND
            ${PYCLIF}
            --matcher_bin=${CLIF_MATCHER}
            --ccdeps_out ${gen_cc} --header_out ${gen_h} --ccinit_out ${gen_init}
            --modname=${module_name}
            -I${CMAKE_BINARY_DIR}/kaldi
            -I${CMAKE_SOURCE_DIR}/kaldi
            "-f-I${PYTHON_INCLUDE_DIRS} \
               -I${CLIF_DIR}/.. \
               -I${KALDI_SRC_DIR} \
               -I${CMAKE_SOURCE_DIR}/kaldi \
               -I.. \
               -I${OPENFST_INC_DIR} \
               -DKALDI_DOUBLEPRECISION=${DOUBLE_PRECISION} \
               -DHAVE_EXECINFO_H=1 \
               -DHAVE_CXXABI_H \
               -DHAVE_ATLAS \
               -I${ATLAS_INC_DIR} \
               -msse -msse2 -pthread -g -O0 \
               -DKALDI_PARANOID \
               -fPIC \
               -std=c++11 \
                 ${CMAKE_CXX_FLAGS}"
            ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
        COMMAND
            ${CMAKE_SOURCE_DIR}/add_namespace.sh ${gen_cc} ${PYCLIF_LIBRARY_NAMESPACES}
        VERBATIM
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file} ${PYCLIF_LIBRARY_CC_DEPS} ${PYCLIF_LIBRARY_CLIF_DEPS}
    )
    add_custom_target(
      ${name} ALL
      DEPENDS ${gen_cc} ${gen_h} ${gen_init}
      VERBATIM
    )
endfunction(add_pyclif_library)


include_directories(${CLIF_DIR}/..)
add_subdirectory("kaldi")
