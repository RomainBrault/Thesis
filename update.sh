set -e

git pull overleaf master
git pull
git add -A
git commit -m"Sync with Overleaf"
git push both