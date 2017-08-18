cmake_minimum_required(VERSION 3.5)

project(Pykaldi)

# Prevent in-source builds
if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
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
set(OPENFST_LIB_DIR ${KALDI_TOOLS_DIR}/openfst/lib)

# Find Numpy
if(NOT NUMPY_INC_DIR)
  message(FATAL_ERROR "You need to specify Numpy include directory. Set -DNUMPY_INC_DIR option.")
endif(NOT NUMPY_INC_DIR)

####################################################################################################
# From: https://github.com/usc-sail/barista/blob/master/CMakeLists.txt
#  Set compiler and linker flags
####################################################################################################
set (CMAKE_CXX_FLAGS
"${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -pthread -fwrapv -Wall -Wno-sign-compare -Wno-write-strings -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -g")

set(CMAKE_INSTALL_RPATH "${KALDI_LIBRARIES_DIR}:${OPENFST_LIB_DIR}:$ORIGIN/..:$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

####################################################################################################
# Add include directories
####################################################################################################
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/kaldi
  ${CMAKE_BINARY_DIR}/kaldi
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
  ${KALDI_LIBRARIES_DIR}
  ${OPENFST_LIB_DIR}
)

####################################################################################################
# Function to set up rules to invoke pyclif on a .clif file
# and generate the wrapper .cc and .h files.
# It also adds a custom target for building the python extension.
#
# Arguments:
#   name - name of the extension
#   pyclif_file - pyclif file to compile into a python extension
#
# Options:
#   LIBRARIES - (multi-value) libraries (e.g. kaldi-matrix, kaldi-base) to link against
#   NAMESPACES - (multi-value) additional namespaces to make visible in the generated .cc files
#   RPATH - colon separated list of additional runtime paths
#   CLIF_DEPS - (multi-value) other clif targets that this target depends on
####################################################################################################
function(add_pyclif_library name pyclif_file)
  set(oneValueArgs RPATH)
  set(multiValueArgs CLIF_DEPS NAMESPACES BASETYPES LIBRARIES)
  cmake_parse_arguments(PYCLIF_LIBRARY "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # PYCLIF configuration
  string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
  set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.cc")
  set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.h")
  set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap-init.cc")
  string(REPLACE "-" "_" module_name ${name})

  # This defines the call to pyclif
  add_custom_command(
      OUTPUT ${gen_cc} ${gen_h} ${gen_init}
      COMMAND
          ${PYCLIF}
          --matcher_bin=${CLIF_MATCHER}
          --ccdeps_out ${gen_cc} --header_out ${gen_h} --ccinit_out ${gen_init}
          --modname=${module_name}
          -I${CMAKE_SOURCE_DIR}/kaldi
          -I${CMAKE_BINARY_DIR}/kaldi
          "-f-I${PYTHON_INCLUDE_DIRS} \
             -I${CMAKE_SOURCE_DIR} \
             -I${CMAKE_SOURCE_DIR}/kaldi \
             -I${CMAKE_BINARY_DIR}/kaldi \
             -I${KALDI_SRC_DIR} \
             -I${OPENFST_INC_DIR} \
             -I${ATLAS_INC_DIR} \
             ${CLIF_CXX_FLAGS} \
             ${CMAKE_CXX_FLAGS}"
          ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
      COMMAND
          ${CMAKE_SOURCE_DIR}/use_namespace.sh ${gen_cc} ${PYCLIF_LIBRARY_NAMESPACES}
      COMMAND
          ${CMAKE_SOURCE_DIR}/add_namespace.sh ${gen_cc} ${PYCLIF_LIBRARY_BASETYPES}
      COMMAND
          ${CMAKE_SOURCE_DIR}/extern_base.sh ${gen_h} ${PYCLIF_LIBRARY_BASETYPES}
      VERBATIM
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
  )

  # Create a custom target to run pyclif whenever the file changes
  add_custom_target(
    ${module_name}_pyclif
    ALL
    DEPENDS ${gen_cc} ${gen_h} ${gen_init}
    VERBATIM
  )

  # Add dependencies on the other pyclif targets
  foreach(DEP ${PYCLIF_LIBRARY_CLIF_DEPS})
    add_dependencies(${module_name}_pyclif ${DEP})
  endforeach(DEP)

  # Manage CC files through the correct function
  # PYCLIFDEP flag adds a dependency on ${module_name}_pyclif target
  # Pass along any arguments sent to this
  add_CC_library(${module_name}
                 "${gen_cc};${gen_init}"
                 PYCLIFDEP
                 RPATH ${PYCLIF_LIBRARY_RPATH}
                 CLIF_DEPS ${PYCLIF_LIBRARY_CLIF_DEPS}
                 LIBRARIES ${PYCLIF_LIBRARY_LIBRARIES})

endfunction(add_pyclif_library)

####################################################################################################
# This function handles the compilation of python extension modules.
#
# arguments:
#   module_name - name of the python extension module
#   SOURCES - (multi-value) list of source files to compile into a shared library
#
# options:
#   OUTPUTDIR - Overrides the output directory for shared libraries
#   PYCLIFDEP - (flag) add a dependency on ${module_name}-pyclif target. Should be set for files generated with pyclif
#   LIBRARIES - (multi-value) libraries (e.g. kaldi-matrix, kaldi-base) to link against
#   RPATH - colon separated list of additional runtime paths
#   CLIF_DEPS - (multi-value) other clif targets that this target depends on
####################################################################################################
function(add_CC_library module_name SOURCES)
  set(options PYCLIFDEP)
  set(multiValueArgs LIBRARIES CLIF_DEPS)
  set(oneValueArgs RPATH OUTPUTDIR)
  cmake_parse_arguments(PYCLIF_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(REPLACE "-" "_" module_name ${module_name})
  if("${PYCLIF_LIBRARY_OUTPUTDIR}" STREQUAL "")
      string(REPLACE "/build" "/build/lib" PYCLIF_LIBRARY_OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  message("Module: ${module_name}")

  message("- Sources:")

  foreach(source ${SOURCES})
    message("    ${source}")
  endforeach()

  # Create extension module
  add_library(${module_name} SHARED ${SOURCES})

  message("- Dependencies:")

  # If PYCLIFDEP is set, add a dependency on ${module_name}_pyclif
  if(PYCLIF_LIBRARY_PYCLIFDEP)
    message("    ${module_name}_pyclif")
    add_dependencies(${module_name} ${module_name}_pyclif)
  endif(PYCLIF_LIBRARY_PYCLIFDEP)

  # Link with other extension modules
  foreach(LIB ${PYCLIF_LIBRARY_CLIF_DEPS})
    message("    ${LIB}")
    target_link_libraries(${module_name} ${LIB})
  endforeach(LIB)

  message("- Libraries:")

  # Link with clif library
  if(NOT "${module_name}" STREQUAL "${CLIF_LIBRARY}")
    message("    ${CLIF_LIBRARY}")
    target_link_libraries(${module_name} ${CLIF_LIBRARY})
  endif()

  # Link with other libraries
  foreach(LIB ${PYCLIF_LIBRARY_LIBRARIES})
    message("    ${LIB}")
    target_link_libraries(${module_name} ${LIB})
  endforeach(LIB)

  message("- Runtime path: ${PYCLIF_LIBRARY_RPATH}")

  message("- Output directory: ${PYCLIF_LIBRARY_OUTPUTDIR}")

  # Do not prepend extension modules with "lib"
  set_target_properties(${module_name} PROPERTIES PREFIX "")

  # Set the output directory for the extension module
  set_target_properties(${module_name}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR}
    LIBRARY_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR}
    RUNTIME_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR}
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${PYCLIF_LIBRARY_RPATH}")

  message("")

endfunction(add_CC_library)


####################################################################################################
# Add target for clif library
####################################################################################################
set(CLIF_LIBRARY "_clif")
string(REPLACE "/build" "/build/lib/kaldi" CLIF_LIB_LOCATION ${CMAKE_CURRENT_BINARY_DIR})
add_CC_library(${CLIF_LIBRARY}
  "${CMAKE_SOURCE_DIR}/clif/python/optional.cc;${CMAKE_SOURCE_DIR}/clif/python/runtime.cc;${CMAKE_SOURCE_DIR}/clif/python/slots.cc;${CMAKE_SOURCE_DIR}/clif/python/types.cc"
  OUTPUTDIR ${CLIF_LIB_LOCATION}
)

add_subdirectory("kaldi")
