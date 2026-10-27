#!/usr/bin/env bash
# Compile every TikZ diagram in docs/diagrams/ (except the shared style.tex)
# with pdflatex and write docs/assets/diagrams/<name>.svg.
# Needs pdflatex (TeX Live with pgf, standalone and helvet) and poppler's
# pdftocairo. Adapted from epiforecasts/BVDOutbreakSize.
#
#   scripts/build_diagrams.sh          # every diagram
#   scripts/build_diagrams.sh --png    # also write 150 dpi PNG previews
#                                      # to output/diagrams/
#
# The SVGs are committed, since CI has no LaTeX.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="$repo_root/docs/diagrams"
out_dir="$repo_root/docs/assets/diagrams"
png_dir="$repo_root/output/diagrams"
png=false
[[ "${1:-}" == "--png" ]] && png=true

for tool in pdflatex pdftocairo; do
  command -v "$tool" >/dev/null || {
    echo "build_diagrams: $tool is not on PATH" >&2
    exit 1
  }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/diagrams.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out_dir"

shopt -s nullglob
for tex in "$src_dir"/*.tex; do
  name="$(basename "$tex" .tex)"
  [[ "$name" == "style" ]] && continue
  echo "build_diagrams: $name"
  (
    cd "$src_dir"
    pdflatex -interaction=nonstopmode -halt-on-error \
      -output-directory "$tmp" "$tex" >"$tmp/$name.out" 2>&1
  ) || {
    tail -n 30 "$tmp/$name.out" >&2
    exit 1
  }
  pdftocairo -svg "$tmp/$name.pdf" "$out_dir/$name.svg"
  if $png; then
    mkdir -p "$png_dir"
    pdftocairo -png -r 150 -singlefile "$tmp/$name.pdf" "$png_dir/$name"
  fi
done
