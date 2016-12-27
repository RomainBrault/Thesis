set -e

git pull github master
rm -rf build || true
mkdir build
cd build
cmake ..
make -j1
