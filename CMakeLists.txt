PROJECT(ThesisRomainBrault NONE)
cmake_minimum_required(VERSION 2.8)
SET(LATEX_OUTPUT_PATH build)
SET(PDFLATEX_COMPILER xelatex)
SET(LATEX_COMPILER_FLAGS -synctex=1 -interaction=nonstopmode)
INCLUDE(CMakeLatex/UseLATEX.cmake)

file(GLOB_RECURSE contents_files RELATIVE ${CMAKE_SOURCE_DIR} *.tex)

ADD_LATEX_DOCUMENT(
    ThesisRomainBrault.tex
    INPUTS ${contents_files}
    IMAGE_DIRS gfx
    BIBFILES Bibliography.bib
    USE_GLOSSARY
    )
