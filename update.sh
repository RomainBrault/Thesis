set -e

git pull github master
git add -A
git commit -m"Sync with Overleaf"
git push both


git pull github master
git add -A
git commit -m"Sync with Overleaf"
git push both
