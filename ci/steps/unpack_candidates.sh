#!/usr/bin/env bash
# actions/download-artifact puts each `candidate-<id>` artefact in a
# directory of that name. Rename them to `<id>`, which is the layout the
# select step and run_dry.sh use.
# shellcheck source=ci/steps/lib.sh
source "$(dirname "$0")/lib.sh"

shopt -s nullglob
for d in "$WORK"/candidates/candidate-*; do
  id=${d##*/candidate-}
  rm -rf "$WORK/candidates/$id"
  mv "$d" "$WORK/candidates/$id"
done
ls "$WORK/candidates"
