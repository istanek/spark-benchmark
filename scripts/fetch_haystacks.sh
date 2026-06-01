#!/usr/bin/env bash
# scripts/fetch_haystacks.sh — download the public-domain haystack corpora
# used by the long_context_retrieval suite.
#
# The texts are large (Project Gutenberg books) and are deliberately NOT
# committed to the repo (see .gitignore). Run this once before running the
# long-context suite. Re-running is idempotent: existing files are kept
# unless --force is passed.
#
# Usage:
#   scripts/fetch_haystacks.sh            # fetch any missing haystacks
#   scripts/fetch_haystacks.sh --force    # re-download everything
#
# Requirements: curl (or wget), sha256sum (optional, for checksum reporting)

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly DEST="${REPO_ROOT}/data/long_context/haystacks"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
fail() { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
  || fail "need curl or wget on PATH"

mkdir -p "$DEST"

# name | url
read -r -d '' HAYSTACKS <<'EOF' || true
melville_moby_dick.txt|https://www.gutenberg.org/files/2701/2701-0.txt
darwin_origin_of_species.txt|https://www.gutenberg.org/files/1228/1228-0.txt
EOF

fetch_one() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$out" "$url"
  else
    wget -q -O "$out" "$url"
  fi
}

while IFS='|' read -r name url; do
  [[ -z "$name" ]] && continue
  out="${DEST}/${name}"
  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    log "exists, skipping: $name (use --force to re-download)"
  else
    log "fetching $name"
    fetch_one "$url" "$out" || fail "download failed: $url"
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sum="$(sha256sum "$out" | awk '{print $1}')"
    bytes="$(wc -c < "$out" | tr -d ' ')"
    log "  -> ${bytes} bytes, sha256 ${sum}"
  fi
done <<< "$HAYSTACKS"

log "done. haystacks in: $DEST"
