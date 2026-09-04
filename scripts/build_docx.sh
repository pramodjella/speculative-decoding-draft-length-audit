#!/usr/bin/env bash
# Word export of the paper, for reviewers who want to comment in Word.
#
# The arXiv deliverable is paper/paper.pdf built from paper/paper.tex; this .docx is a
# convenience copy and is NOT the source of truth. Regenerate it whenever paper.tex changes,
# or it silently ships stale numbers (it did, by one review round).
#
# pandoc does not understand IEEEtran's \IEEEauthorblockN, so the author list is dropped from
# the converted body and has to be supplied as metadata explicitly -- otherwise the .docx
# goes out with a title and no authors. The affiliation rides as a trailing author line
# because pandoc's docx writer ignores `institute`.
set -eu
cd "$(dirname "$0")/.."

# Resolve to an absolute path before testing it. `command -v pandoc` succeeding only means
# pandoc is on PATH -- it leaves $PANDOC as the bare word, and `[ -x pandoc ]` then tests the
# relative path ./pandoc and fails. Take command -v's answer rather than the input.
PANDOC="${PANDOC:-}"
if [ -z "$PANDOC" ]; then
  PANDOC="$(command -v pandoc 2>/dev/null || true)"
fi
if [ -z "$PANDOC" ] || [ ! -x "$PANDOC" ]; then
  for c in "$HOME/AppData/Local/Pandoc/pandoc.exe" "/c/Program Files/Pandoc/pandoc.exe"; do
    [ -x "$c" ] && PANDOC="$c" && break
  done
fi
[ -n "$PANDOC" ] && [ -x "$PANDOC" ] || {
  echo "pandoc not found; set PANDOC=/path/to/pandoc" >&2; exit 1; }

bash scripts/check_stale.sh

cd paper
"$PANDOC" paper.tex -o paper.docx \
  --resource-path=. \
  --metadata title="How Much of Speculative Decoding's Adaptive Draft-Length Headroom Is Real? An Audit Under Tuned Baselines and Paired Wall-Clock Protocols" \
  --metadata author="Pramod Kumar Reddy Jella" \
  --metadata author="Dr. Yash Dixit" \
  --metadata author="Naman Dwivedi" \
  --metadata author="Dr. Raj Dandekar" \
  --metadata author="Dr. Rajat Dandekar" \
  --metadata author="Dr. Sreedath Panat" \
  --metadata author="Vizuara AI Labs, Inference Engineering Research Track — hello@vizuara.com"

echo "wrote paper/paper.docx"
