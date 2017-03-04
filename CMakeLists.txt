project(ThesisRomainBrault NONE)
cmake_minimum_required(VERSION 2.8)

message(STATUS "LATEX_OUTPUT_PATH: " ${LATEX_OUTPUT_PATH})
message(STATUS "PDFLATEX_COMPILER: " ${PDFLATEX_COMPILER})
message(STATUS "LATEX_USE_SYNCTEX: " ${LATEX_USE_SYNCTEX})

set(LATEX_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/${LATEX_OUTPUT_PATH}"
    CACHE STRING "Output directory")
set(PDFLATEX_COMPILER ${PDFLATEX_COMPILER}
    CACHE STRING "Latex compiler")
set(LATEX_COMPILER_FLAGS "-file-line-error -shell-escape -output-driver=\"xdvipdfmx -z 0\" -bite"
    CACHE STRING "Flags passed to latex")
set(LATEX_USE_SYNCTEX ${LATEX_USE_SYNCTEX}
    CACHE STRING "Use Synctex")
set(PYTHONTEX_COMPILER pythontex
    CACHE STRING "Pythontex compiler")

include(${CMAKE_SOURCE_DIR}/CMakeLatex/UseLATEX.cmake)

file(GLOB_RECURSE contents_files RELATIVE ${CMAKE_SOURCE_DIR} *.tex
     EXCLUDE bin/ Users/)

add_latex_document(
    ThesisRomainBrault.tex
    INPUTS ${contents_files}
    IMAGE_DIRS gfx
    BIBFILES Bibliography.bib)
