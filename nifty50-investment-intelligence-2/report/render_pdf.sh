#!/usr/bin/env bash
# Render report/report.html -> report/NIFTY50_Investment_Intelligence_Report.pdf
# Requires Google Chrome (headless). Run from the project root.
set -euo pipefail

HTML="${1:-report/report.html}"
PDF="${2:-report/NIFTY50_Investment_Intelligence_Report.pdf}"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || CHROME="$(command -v google-chrome || command -v chromium || true)"
[ -n "$CHROME" ] || { echo "Chrome/Chromium not found"; exit 1; }

ABS_HTML="file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=12000 \
  --print-to-pdf="$PDF" "$ABS_HTML" 2>/dev/null

echo "wrote $PDF"
