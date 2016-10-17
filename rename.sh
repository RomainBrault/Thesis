set -e

git pull overleaf master
git pull
rm -rf build || true
mkdir build
cd build
cmake ..
make -j8
