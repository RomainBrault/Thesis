set -e

git pull overleaf master
git pull
rm -rf build || true
mkdir build
cd build
cmake ..
make -j8
mv ThesisRomainBrault.pdf ..
cd ..
git add -A
git commit -m"Local make"
git push both