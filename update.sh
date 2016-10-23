set -e

git pull overleaf master
git pull github master
git add -A
git commit -m"Sync with Overleaf"
git push both


git pull overleaf master
git pull github master
git add -A
git commit -m"Sync with Overleaf"
git push both