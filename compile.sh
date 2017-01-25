#!/bin/sh

set -e

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.
UPDATE=false
FORCE=false

while getopts "h?uf" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    u)  UPDATE=true
        ;;
    f)  FORCE=true
        ;;
    esac
done

if [[ ${UPDATE} = true ]]; then
    git pull github master
fi
if [[ ${FORCE} = true ]]; then
    rm -rf build || true
fi
mkdir -p build
cd build
if [[ ${FORCE} = true ]]; then
    cmake ..
    make all
else
    make ThesisRomainBrault
fi
cd ..
