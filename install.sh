set -e

git pull overleaf master
rm -rf build || true
mkdir build
cd build
cmake ..
make -j8
mv ThesisRomainBrault.pdf ..
cd ..
git add -A
git commit -m"Sync with Overleaf"
git push both