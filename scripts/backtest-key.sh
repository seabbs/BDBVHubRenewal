#!/usr/bin/env bash
# Print a hash of what the Pages backtest reads, for its cache key.
#
# Files are hashed by content: a .gz file is read decompressed, so a
# rewrite of the same table with a new gzip time keeps the hash. Python
# caches, the raw downloads in data/raw and data/sources.json (where and
# when they came from) are left out; the backtest reads the tables made
# from them.
set -euo pipefail

find src config uv.lock .github/workflows/pages.yml results data \
  -name __pycache__ -prune -o \
  -path data/raw -prune -o \
  -path data/sources.json -prune -o \
  -type f -print |
  LC_ALL=C sort |
  while read -r f; do
    printf '%s ' "$f"
    case "$f" in
      *.gz) gzip -dc "$f" ;;
      *) cat "$f" ;;
    esac | sha256sum | cut -d' ' -f1
  done |
  sha256sum | cut -d' ' -f1
