project(ThesisRomainBrault NONE)
cmake_minimum_required(VERSION 2.8)

message(STATUS "LATEX_OUTPUT_PATH: " ${LATEX_OUTPUT_PATH})
message(STATUS "PDFLATEX_COMPILER: " ${PDFLATEX_COMPILER})
message(STATUS "LATEX_USE_SYNCTEX: " ${LATEX_USE_SYNCTEX})

set(LATEX_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/${LATEX_OUTPUT_PATH}"
    CACHE STRING "Output directory")
set(PDFLATEX_COMPILER ${PDFLATEX_COMPILER}
    CACHE STRING "Latex compiler")
set(LATEX_COMPILER_FLAGS 
    "-shell-escape -file-line-error -interaction=nonstopmode" 
    CACHE STRING "Flags passed to latex")
separate_arguments(LATEX_COMPILER_FLAGS)
message(STATUS ${LATEX_COMPILER_FLAGS})
set(LATEX_USE_SYNCTEX ${LATEX_USE_SYNCTEX}
    CACHE STRING "Use Synctex")
set(PYTHONTEX_COMPILER pythontex
    CACHE STRING "Pythontex compiler")

include(${CMAKE_SOURCE_DIR}/CMakeLatex/UseLATEX.cmake)

file(GLOB_RECURSE contents_files RELATIVE 
     ${CMAKE_SOURCE_DIR} *.tex *.icc *.tikz *.pgf *.otf *.ttc)
foreach(item ${contents_files})
  if((${item} MATCHES "bin/*") OR
     (${item} MATCHES "build/*") OR
     (${item} MATCHES ".git/*"))
    list(REMOVE_ITEM contents_files ${item})
  endif()
endforeach()
message(STATUS "contents_files: " ${contents_files})

add_latex_document(ThesisRomainBrault.tex
    INPUTS ${contents_files}
    CONFIGURE classicthesis-config.tex
    IMAGE_DIRS gfx
    BIBFILES Bibliography.bib
    FORCE_PDF)
