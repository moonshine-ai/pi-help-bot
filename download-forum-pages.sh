#!/usr/bin/env bash
# Download forum pages from Raspberry Pi forums, paginating by start= (step 25, max 46008).
# Waits 3-5 seconds between fetches to be polite to the server.
# Uses Playwright (via playwright_fetch.py) for headless fetch so we get rendered content.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://forums.raspberrypi.com/viewforum.php?f=91"
STEP=25
MAX_START=46008
OUTPUT_DIR="${1:-./forum_pages}"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

for (( start = 0; start <= MAX_START; start += STEP )); do
  url="${BASE_URL}&start=${start}"
  outfile="viewforum_f91_start_${start}.html"
  echo "Fetching start=${start} -> ${outfile}"
  python "$SCRIPT_DIR/playwright_fetch.py" "$url" "$outfile" \
    --har "./har_${start}.har" \
    --screenshot "./screenshot_${start}.png"
  if (( start < MAX_START )); then
    delay=$(( 3 + RANDOM % 3 ))
    echo "  sleeping ${delay}s..."
    sleep "$delay"
  fi
done

echo "Done. Pages saved in ${OUTPUT_DIR}"
