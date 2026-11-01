#!/usr/bin/env bash
# Run this on your Mac inside your empty GitHub clone (Cursor integrated terminal).
# Requires: Origin CLI installed and logged in (`origin auth login`).

set -euo pipefail

ORIGIN_REPO="davism22/cps493-emissions-disclosure"
TMPDIR="${TMPDIR:-/tmp}/cps493-origin-clone-$$"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Error: run this from the root of your GitHub clone (must be a git repo)."
  exit 1
fi

if ! command -v origin >/dev/null 2>&1; then
  echo "Install Origin CLI: curl -fsSL https://downloads.cursor.com/origin/install.sh | sh"
  exit 1
fi

echo "Cloning Origin repo ${ORIGIN_REPO} ..."
origin repo clone "${ORIGIN_REPO}" "${TMPDIR}"

echo "Copying project files into this folder (keeping your GitHub remote) ..."
rsync -a --exclude .git --exclude .venv --exclude __pycache__ "${TMPDIR}/" ./

rm -rf "${TMPDIR}"

git add -A
if git diff --staged --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "Import CPS493 emissions ML project from Origin"
fi

echo "Pushing to GitHub ..."
git push origin main

echo "Done. Your GitHub repo should now have the full codebase."
