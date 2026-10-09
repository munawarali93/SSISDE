#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p .build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build double_well_changes.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build double_well_changes.tex
cp .build/double_well_changes.pdf double_well_changes.pdf
