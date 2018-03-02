#####################################################################
# Provides a target for the documentation using Cmake
# Build the documentation using `make docs`
# Sources: https://eb2.co/blog/2012/03/sphinx-and-cmake-beautiful-documentation-for-c-projects/
#####################################################################
#if(NOT DEFINED SPHINX_THEME)
#    set(SPHINX_THEME default)
#endif()

#if(NOT DEFINED SPHINX_THEME_DIR)
#    set(SPHINX_THEME_DIR)
#endif()

# configured documentation tools and intermediate build results
#set(BINARY_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_build")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/html")



# call autogen everytime the readme is updated
# set(AUTOGEN_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/_autosummary")
# set(AUTOGEN_IN "API.rst")
# add_custom_command(
#   OUTPUT ${AUTOGEN_OUTDIR}
#   COMMAND rm -rf ${AUTOGEN_OUTDIR}
#   COMMAND sphinx-autogen ${AUTOGEN_IN}
#                --imported-members
#                -o ${AUTOGEN_OUTDIR}
#   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#   DEPENDS ${AUTOGEN_IN}
#   COMMENT "Calling autogen"
#   VERBATIM
# )

add_custom_target(docs
  COMMAND  python -msphinx
      -q -b html
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${SPHINX_HTML_DIR}"
  VERBATIM
  COMMENT "Building HTML documentation with Sphinx"
  # DEPENDS ${AUTOGEN_OUTDIR}
)
