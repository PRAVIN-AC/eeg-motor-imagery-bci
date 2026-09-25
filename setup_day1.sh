#!/usr/bin/env bash
# Day 1 — Environment setup for the EEG Motor-Imagery BCI project.
# Usage: bash setup_day1.sh
set -e

echo "== Creating virtual environment (.venv) =="
python3 -m venv .venv
source .venv/bin/activate

echo "== Upgrading pip =="
pip install --upgrade pip

echo "== Installing project dependencies =="
pip install -r requirements.txt

echo "== Running environment check =="
python src/check_env.py

echo ""
echo "== Git =="
if [ ! -d ".git" ]; then
  git init
  git add .
  git commit -m "Day 1: environment setup, project scaffold"
  echo "Local repo initialized and first commit made."
  echo "Now create an empty repo on GitHub and run:"
  echo "  git remote add origin <your-repo-url>"
  echo "  git branch -M main"
  echo "  git push -u origin main"
else
  echo "Git repo already initialized — commit manually when ready."
fi
