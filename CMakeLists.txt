cmake_minimum_required(VERSION 3.5)

project(PyKaldi)

# Prevent in-source builds
if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

####################################################################################################
# Find dependencies
####################################################################################################

# Find Python
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
get_filename_component(PYTHON_LIB_DIR "${PYTHON_LIBRARIES}" PATH)

# Find Numpy
if(NOT NUMPY_INC_DIR)
  message(FATAL_ERROR "You need to specify Numpy include directory. Set -DNUMPY_INC_DIR option.")
endif(NOT NUMPY_INC_DIR)

# Find CLIF
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

if(NOT CLIF_MATCHER)
  set(CLIF_MATCHER $ENV{CLIF_MATCHER})
  if (NOT CLIF_MATCHER)
    get_filename_component(_Python_PREFIX "${PYTHON_LIB_DIR}" PATH)
    find_program(CLIF_MATCHER clif-matcher PATH ${_Python_PREFIX}/clang/bin)
    unset(_Python_PREFIX)
    if(NOT CLIF_MATCHER)
      message(FATAL_ERROR "Could not find 'clif-matcher'. Set CLIF_MATCHER environment variable or -DCLIF_MATCHER option.")
    endif(NOT CLIF_MATCHER)
  endif(NOT CLIF_MATCHER)
endif(NOT CLIF_MATCHER)
message("Using CLIF_MATCHER: ${CLIF_MATCHER}")

# Find Kaldi
if(NOT KALDI_DIR)
  set(KALDI_DIR $ENV{KALDI_DIR})
  if(NOT KALDI_DIR)
    message(FATAL_ERROR "You need to specify Kaldi directory. Set KALDI_DIR environment variable or -DKALDI_DIR option.")
  endif(NOT KALDI_DIR)
endif(NOT KALDI_DIR)

set(KALDI_SRC_DIR "${KALDI_DIR}/src")
set(KALDI_TOOLS_DIR "${KALDI_DIR}/tools")
set(KALDI_LIBRARIES_DIR "${KALDI_SRC_DIR}/lib")
set(OPENFST_LIB_DIR "${KALDI_TOOLS_DIR}/openfst/lib")

####################################################################################################
#  Set compiler and linker flags
####################################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS} ${LD_LIBS}")
if(CUDA)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${CUDA_LD_FLAGS} ${CUDA_LD_LIBS}")
endif(CUDA)

set(CMAKE_INSTALL_RPATH "${PYTHON_LIB_DIR};${KALDI_LIBRARIES_DIR};${OPENFST_LIB_DIR}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(ORIGIN "@loader_path")
else()
  set(ORIGIN "$ORIGIN")
endif()
file(GLOB children RELATIVE ${CMAKE_SOURCE_DIR}/kaldi ${CMAKE_SOURCE_DIR}/kaldi/*)
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/kaldi/${child})
    LIST(APPEND CMAKE_INSTALL_RPATH "${ORIGIN}/../${child}")
  endif()
endforeach()
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

include_directories(
  ${CMAKE_SOURCE_DIR}/kaldi/lib
  ${CMAKE_SOURCE_DIR}/kaldi
  ${CMAKE_BINARY_DIR}/kaldi
  ${KALDI_SRC_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INC_DIR}
)

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
  set(multiValueArgs CLIF_DEPS NAMESPACES LIBRARIES)
  cmake_parse_arguments(PYCLIF_LIBRARY "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # PYCLIF configuration
  string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
  set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.cc")
  set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap.h")
  set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}-clifwrap-init.cc")
  string(REPLACE "-" "_" module_name ${name})

  if(PYTHON_VERSION_MAJOR EQUAL 3)
    set(PYOUTPUT "--py3output")
  else()
    set(PYOUTPUT "--py2output")
  endif()

  # This defines the call to pyclif
  add_custom_command(
      OUTPUT ${gen_cc} ${gen_h} ${gen_init}
      COMMAND
          ${PYCLIF}
          ${PYOUTPUT}
          --matcher_bin=${CLIF_MATCHER}
          --ccdeps_out ${gen_cc} --header_out ${gen_h} --ccinit_out ${gen_init}
          --modname=${PACKAGE}.${module_name}
          --prepend=clif/python/types.h
          -I${CMAKE_SOURCE_DIR}/kaldi/lib
          -I${CMAKE_SOURCE_DIR}/kaldi
          -I${CMAKE_BINARY_DIR}/kaldi
          -I${KALDI_SRC_DIR}
          "-f-I${PYTHON_INCLUDE_DIRS} \
             -I${CMAKE_SOURCE_DIR}/kaldi/lib \
             -I${CMAKE_SOURCE_DIR}/kaldi \
             -I${CMAKE_BINARY_DIR}/kaldi \
             -I${KALDI_SRC_DIR} \
             ${CLIF_CXX_FLAGS} \
             ${CMAKE_CXX_FLAGS}"
          ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
      COMMAND
          ${CMAKE_SOURCE_DIR}/tools/use_namespace.sh ${gen_cc} ${PYCLIF_LIBRARY_NAMESPACES}
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

  # Call the custom function for building the extension module
  # PYCLIFDEP flag adds a dependency on ${module_name}_pyclif target
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

  if(DEBUG)
    message("Module: ${module_name}")
    message("- Sources:")
    foreach(source ${SOURCES})
      message("    ${source}")
    endforeach()
  endif(DEBUG)

  # Create extension module
  add_library(${module_name} SHARED ${SOURCES})
  set_target_properties(${module_name} PROPERTIES SUFFIX ".so")

  if(DEBUG)
    message("- Dependencies:")
  endif(DEBUG)

  # If PYCLIFDEP is set, add a dependency on ${module_name}_pyclif
  if(PYCLIF_LIBRARY_PYCLIFDEP)
    if(DEBUG)
      message("    ${module_name}_pyclif")
    endif(DEBUG)
    add_dependencies(${module_name} ${module_name}_pyclif)
  endif(PYCLIF_LIBRARY_PYCLIFDEP)

  # Link with other extension modules
  foreach(LIB ${PYCLIF_LIBRARY_CLIF_DEPS})
    if(DEBUG)
      message("    ${LIB}")
    endif(DEBUG)
    target_link_libraries(${module_name} ${LIB})
  endforeach(LIB)

  if(DEBUG)
    message("- Libraries:")
  endif(DEBUG)

  # Link with clif library
  if(NOT "${module_name}" STREQUAL "${CLIF_LIBRARY}")
    if(DEBUG)
      message("    ${CLIF_LIBRARY}")
    endif(DEBUG)
    target_link_libraries(${module_name} ${CLIF_LIBRARY})
  endif()

  # Link with kaldi libraries
  foreach(LIB ${PYCLIF_LIBRARY_LIBRARIES})
    if(DEBUG)
      message("    ${LIB}")
    endif(DEBUG)
    target_link_libraries(${module_name} ${LIB})
  endforeach(LIB)

  # Do not prepend extension modules with "lib"
  set_target_properties(${module_name} PROPERTIES PREFIX "")

  # Set the output directory for the extension module
  set_target_properties(${module_name}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR}
    LIBRARY_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR}
    RUNTIME_OUTPUT_DIRECTORY ${PYCLIF_LIBRARY_OUTPUTDIR})

  if(DEBUG)
    message("- Runtime path: ${PYCLIF_LIBRARY_RPATH}")
    message("- Output directory: ${PYCLIF_LIBRARY_OUTPUTDIR}")
    message("")
  endif(DEBUG)

endfunction(add_CC_library)

####################################################################################################
# Add target for clif library
####################################################################################################

set(CLIF_LIBRARY "_clif")
string(REPLACE "/build" "/build/lib/kaldi/lib" CLIF_LIB_OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
set(CLIF_SRC_DIR "${CMAKE_SOURCE_DIR}/kaldi/lib")

if(DEBUG)
  add_CC_library(${CLIF_LIBRARY}
    "${CLIF_SRC_DIR}/clif/python/optional.cc;${CLIF_SRC_DIR}/clif/python/runtime.cc;${CLIF_SRC_DIR}/clif/python/slots.cc;${CLIF_SRC_DIR}/clif/python/types.cc"
    OUTPUTDIR ${CLIF_LIB_OUTPUTDIR}
    LIBRARIES ${PYTHON_LIBRARIES}
  )
else(DEBUG)
  add_CC_library(${CLIF_LIBRARY}
    "${CLIF_SRC_DIR}/clif/python/optional.cc;${CLIF_SRC_DIR}/clif/python/runtime.cc;${CLIF_SRC_DIR}/clif/python/slots.cc;${CLIF_SRC_DIR}/clif/python/types.cc"
    OUTPUTDIR ${CLIF_LIB_OUTPUTDIR}
  )
endif(DEBUG)

####################################################################################################
# Add targets for pykaldi libraries
####################################################################################################

add_subdirectory("kaldi")
add_subdirectory("docs")
