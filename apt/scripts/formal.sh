#!/usr/bin/env bash
# Run every formal proof (SymbiYosys + z3).  Needs yosys, yosys-smtbmc, yosys-witness, sby, z3 on PATH
# (or yowasp-sby with shims named yosys*/ on PATH).  Usage: scripts/formal.sh [sby-binary]
set -euo pipefail
cd "$(dirname "$0")/../formal"
SBY=${1:-${SBY:-sby}}
for f in *.sby; do
  echo "== $f"
  "$SBY" -f "$f" prove
done
