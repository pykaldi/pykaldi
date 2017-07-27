cmake_minimum_required(VERSION 3.5)

project(Pykaldi)

# Prevent in-source builds
if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

# Additional files used by cmake
# TODO: Add scripts FindNumPy and FindKaldi
# set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

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

# Find Kaldi and its dependencies (Atlas, OpenFst, portaudio)
if(NOT KALDI_DIR)
  set(KALDI_DIR $ENV{KALDI_DIR})
  if(NOT KALDI_DIR)
    message(FATAL_ERROR "You need to specify Kaldi directory. Set KALDI_DIR environmen variable or -DKALDI_DIR option.")
  endif(NOT KALDI_DIR)
endif(NOT KALDI_DIR)

set(KALDI_SRC_DIR "${KALDI_DIR}/src")
set(KALDI_TOOLS_DIR "${KALDI_DIR}/tools")
set(KALDI_LIBRARIES_DIR "${KALDI_SRC_DIR}/lib")
set(ATLAS_INC_DIR ${KALDI_TOOLS_DIR}/ATLAS/include)
set(OPENFST_INC_DIR ${KALDI_TOOLS_DIR}/openfst/include)


# Find Numpy
if(NOT NUMPY_INC_DIR)
  message(FATAL_ERROR "You need to specify Numpy include directory. Set -DNUMPY_INC_DIR option.")
endif(NOT NUMPY_INC_DIR)

####################################################################################################
# From: https://github.com/usc-sail/barista/blob/master/CMakeLists.txt
#  Set compiler and linker flags
####################################################################################################
set (CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -msse -msse2 -fPIC -Wall -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -DKALDI_PARANOID -Wno-unused-local-typedefs -Wno-write-strings -Wno-sign-compare -flax-vector-conversions -fpermissive -g -std=c++11")

####################################################################################################
# Add include directories
####################################################################################################
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/kaldi
  ${CMAKE_SOURCE_DIR}/kaldi
  ${CLIF_DIR}/..
  ${KALDI_SRC_DIR}
  ${OPENFST_INC_DIR}
  ${ATLAS_INC_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INC_DIR}
)

####################################################################################################
# Add libraries directories
####################################################################################################
link_directories(
  ${CMAKE_BINARY_DIR}/lib/kaldi
  ${KALDI_LIBRARIES_DIR}
)

####################################################################################################
# Add target for clif libraries
####################################################################################################
set(CLIF_LIBRARIES "_clif")
add_library(${CLIF_LIBRARIES} SHARED
  ${CLIF_DIR}/python/runtime.cc
  ${CLIF_DIR}/python/slots.cc
  ${CLIF_DIR}/python/types.cc
)

# don't prepend wrapper library name with lib
set_target_properties(${CLIF_LIBRARIES} PROPERTIES PREFIX "")

# Set correct location for output
set_target_properties(${CLIF_LIBRARIES}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/kaldi
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/kaldi
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/kaldi
)

####################################################################################################
# Function to set up rules to invoke pyclif on a .clif file
# and generate the wrapper .cc and .h files.
# It also adds a custom target to build the python extension
#
# arguments:
#   name - Name of the extension
#   pyclif_file - Name of the pyclif file to compile
#
# options:
#   LIBRARIES - (multi-value) libraries (e.g. kaldi-matrix, kaldi-base) to link against
#   LIBRARY_DIRS - (multi-value) Currently not used
#   INCLUDE_DIRS - (multi-value) Currently not used
#   CLIF_DEPS - (multi-value) Dependencies on clif targets to add to this target
####################################################################################################
function(add_pyclif_library name pyclif_file)
  set(multiValueArgs LIBRARIES LIBRARY_DIRS INCLUDE_DIRS CLIF_DEPS NAMESPACES)
  cmake_parse_arguments(PYCLIF_LIBRARY "" "" "${multiValueArgs}" ${ARGN})

  # PYCLIF configuration
  string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
  set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.cc")
  set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.h")
  set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap-init.cc")
  string(REPLACE "-" "_" module_name ${pyclif_file_basename})

  # This defines the call to pyclif
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
             -I${ATLAS_INC_DIR} \
             ${CMAKE_CXX_FLAGS}"
          ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
      COMMAND
          ${CMAKE_SOURCE_DIR}/add_namespace.sh ${gen_cc} ${PYCLIF_LIBRARY_NAMESPACES}
      VERBATIM
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
  )

  # Create a custom target to run pyclif whenever the file changes
  add_custom_target(
    ${name}-pyclif
    ALL
    DEPENDS ${gen_cc} ${gen_h} ${gen_init}
    VERBATIM
  )

  # Add dependencies on the other pyclif targets
  foreach(DEP ${PYCLIF_LIBRARY_CLIF_DEPS})
    add_dependencies(${name}-pyclif ${DEP})
  endforeach(DEP)

  # Manage CC files through the correct function
  # PYCLIFDEP flag adds a dependency on ${name}-pyclif target
  # Pass along any arguments sent to this
  add_CC_library(${module_name}
                 ${gen_cc}
                 gen_init ${gen_init}
                 gen_h ${gen_h}
                 PYCLIFDEP
                 ${ARGN})

endfunction(add_pyclif_library)

####################################################################################################
# This function handles the case of .cc files that do not come from pyclif
# It adds a custom target to build the python extension
#
# arguments:
#   extname - Name of the library to build
#   gen_cc - CCP file to compile
#
# options:
#   PYCLIFDEP - (flag) to add a dependency on ${extname}-pyclif target. Should be set for files generated with pyclif
#   LIBRARIES - (multi-value) libraries (e.g. kaldi-matrix, kaldi-base) to link against
#   LIBRARY_DIRS - (multi-value) Currently not used
#   INCLUDE_DIRS - (multi-value) Currently not used
#   EXTENSION_PREFIX - (value) Currently not used
#   EXTENSION_SUFIX - (value) Currently not used
#   CLIF_DEPS - (multi-value) Dependencies on clif targets to add to this target
#   gen_h, gen_init - H and init.CPP files
####################################################################################################
function(add_CC_library extname gen_cc)
  set(options PYCLIFDEP)
  set(multiValueArgs LIBRARIES LIBRARY_DIRS INCLUDE_DIRS CLIF_DEPS)
  set(oneValueArgs gen_h gen_init EXTENSION_PREFIX EXTENSION_SUFIX)
  cmake_parse_arguments(PYCLIF_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # TODO (VM):
  # This might not work in all cases...
  string(REPLACE "pykaldi" "pykaldi/build/lib" output_location ${CMAKE_CURRENT_SOURCE_DIR})

  # Create extension
  add_library(${extname} SHARED ${gen_cc} ${gen_init})

  # Add dependencies
  message("Dependencies for: ${extname}")

  # If PYCLIFDEP is set, add dependency on ${extname}-pyclif
  if(PYCLIF_LIBRARY_PYCLIFDEP)
    message("${name}-pyclif")
    add_dependencies(${extname} ${extname}-pyclif)
  endif(PYCLIF_LIBRARY_PYCLIFDEP)

  # Add dependencies from CLIF_DEPS
  foreach(DEP ${PYCLIF_LIBRARY_CLIF_DEPS})
    message("${DEP}")
    target_link_libraries(${extname} ${DEP})
  endforeach(DEP)

  # Link against CLIF_LIBRARIES
  message("Libraries for: ${extname}")
  message("${CLIF_LIBRARIES}")
  target_link_libraries(${extname} ${CLIF_LIBRARIES})
  foreach(LIB ${PYCLIF_LIBRARY_LIBRARIES})
    message("${LIB}")
    target_link_libraries(${extname} ${LIB})
  endforeach(LIB)

  # don't prepend wrapper library name with lib
  set_target_properties(${extname} PROPERTIES PREFIX "")

  # Create the file in the correct location
  set_target_properties(${extname}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${output_location}
    LIBRARY_OUTPUT_DIRECTORY ${output_location}
    RUNTIME_OUTPUT_DIRECTORY ${output_location})

endfunction(add_CC_library)


include_directories(${CLIF_DIR}/..)
add_subdirectory("kaldi")
