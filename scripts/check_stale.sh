#!/usr/bin/env bash
# Staleness exit-test: fails if any RETIRED number/claim appears in the SHIP SET
# (the documents that leave the repo). docs/CANONICAL.md is the only place retired
# figures may appear (marked as retired). Run before sending/submitting anything.
set -u
cd "$(dirname "$0")/.."

# analyze_make_figures.py is in the ship set because the PNGs it emits go into the arXiv
# tarball: the retired "Bayes(hidden)" rung name survived a passing stale-check by living in
# a figure title rather than in prose, and no amount of grepping the .tex could see it.
SHIP_SET=(paper/paper.tex report/paper_draft.md README.md analyze_make_figures.py)

# Retired claims (see docs/CANONICAL.md): old oracle figures, sigma-multiplier
# language, the pre-audit headline, and the retracted G3 v1/v2 diag claim.
PATTERNS=(
  '24\.9%'
  '21\.5%'
  '[0-9]+[ ]*(σ|sigma)'
  '\+4[–-]*(to|–|-)[ ]*5%[ ]*typical'
  'ship fixed K=2'
  'no controller beats'
  'cheap edits are rare'
  'Bayes\(hidden\)'
  'threshold tuning is exhausted'
  '\+12[^0-9]{0,4}19'
  '\+18\.3'
  '\+4\.3%'
  'pad-verification'
  'padding is ~free at B=1 but'
  'entangles verify-dominance'
  '\+3[^0-9]{0,4}5%'
  'Bayes ?\(hidden\)'
  # Net figures derived from the retired un-paired ladder (bayes_ceiling.json):
  # +3.41/+2.36 became the "+2.3-3.4%" range, and recovery x span gave "+2.2%" for
  # llama8b_reasoning. Corrected paired values are +3.1/+2.0/+2.5.
  '\+2\.3[^0-9]{0,4}3\.4'
  '\+2\.2% net'
  '[0-9]\.[0-9]% net upper bound.*3\.4'
)

fail=0
for f in "${SHIP_SET[@]}"; do
  [ -f "$f" ] || { echo "MISSING ship-set file: $f"; fail=1; continue; }
  for p in "${PATTERNS[@]}"; do
    # lines that explicitly retire a figure are allowed to name it
    hits=$(grep -nE "$p" "$f" | grep -vEi 'historical note|retired|retracted|superseded' || true)
    if [ -n "$hits" ]; then
      echo "STALE CLAIM in $f (pattern: $p):"
      echo "$hits" | head -5
      fail=1
    fi
  done
done

# --- LaTeX structural health (delegated to a real script; embedding escapes in bash
# --- is exactly how the last round of corruption happened) ---
if [ -f paper/paper.tex ]; then
  if ! python scripts/check_latex_health.py paper/paper.tex; then fail=1; fi
fi

if [ "$fail" -eq 0 ]; then
  echo "check_stale: PASS — ship set clean of retired claims"
else
  echo "check_stale: FAIL — fix the above before sending/submitting"
fi
exit $fail
