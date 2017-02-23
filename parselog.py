#!/usr/bin/env python

"""Helper for Xelatex log parsing."""

from pydflatex import LogProcessor

if __name__ == "__main__":
    l = LogProcessor()
    l.process_log('bin/ThesisRomainBrault.log')
