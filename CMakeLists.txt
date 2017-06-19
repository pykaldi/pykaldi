cmake_minimum_required(VERSION 3.5)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

if(NOT PYCLIF)
    find_program(PYCLIF pyclif)
    if(NOT PYCLIF)
        message(FATAL_ERROR "The 'pyclif' program was not found. Specify with -DPYCLIF.")
    endif(NOT PYCLIF)
endif(NOT PYCLIF)
string(REGEX REPLACE "/bin/pyclif$" "" CLIF_INSTALL_DIR ${PYCLIF})

# Function to set up rules to invoke pyclif on a .clif file and generate the wrapper .cc and .h files
function(add_pyclif_library name pyclif_file)
    cmake_parse_arguments(PYCLIF_LIBRARY "" "" "CC_DEPS;CLIF_DEPS;CXX_FLAGS" ${ARGN})

    string(REPLACE ".clif" "" pyclif_file_basename ${pyclif_file})
    set(gen_cc "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}.cc")
    set(gen_h "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}.h")
    set(gen_init "${CMAKE_CURRENT_BINARY_DIR}/${pyclif_file_basename}_init.cc")

    string(REPLACE "-" "_" module_name ${pyclif_file_basename})

    add_custom_target(
        ${name}
        BYPRODUCTS ${gen_cc} ${gen_h} ${gen_init}
        COMMAND
            ${PYCLIF}
            --matcher_bin=/home/victor/clif_backend/build_matcher/bin/clif-matcher
            -c${gen_cc} -g${gen_h} -i${gen_init}
            -I"/home/victor/opt"
            --modname=${module_name}
            "-f-I/usr/include/python2.7 -I/usr/include/c++/tr1/ -I/home/victor/opt/clif -I/home/victor/opt -std=c++11"
            ${CMAKE_CURRENT_SOURCE_DIR}/${pyclif_file}
        VERBATIM
        DEPENDS ${PYCLIF_LIBRARY_CC_DEPS} ${PYCLIF_LIBRARY_CLIF_DEPS}
    )
endfunction(add_pyclif_library)


include_directories("/home/victor/opt/")
add_subdirectory("kaldi")