#!/usr/bin/env bash
# scripts/release.sh — cut a spark-benchmark release
#
# What it does:
#   1. Validates the working tree is clean and we're on main (override with --branch).
#   2. Reads the matching `## [X.Y.Z] - YYYY-MM-DD` section out of CHANGELOG.md.
#   3. Creates an annotated tag `vX.Y.Z` whose message is that section.
#   4. Pushes the tag to origin.
#   5. Creates a GitLab Release whose description is the same section.
#
# Usage:
#   scripts/release.sh 0.2.0                 # cut v0.2.0
#   scripts/release.sh v0.2.0                # same, leading v is allowed
#   scripts/release.sh 0.2.0 --dry-run       # show what would happen, change nothing
#   scripts/release.sh 0.2.0 --branch foo    # allow cutting from a non-main branch
#
# Requirements:
#   - jq, curl, git
#   - GITLAB_TOKEN env var, or a token stored in ~/.git-credentials for gitlab.com
#     (the token must have the `api` scope to create a Release).

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly CHANGELOG="${REPO_ROOT}/CHANGELOG.md"

# --- args --------------------------------------------------------------------

DRY_RUN=0
ALLOWED_BRANCH="main"
VERSION_ARG=""

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        usage 0 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --branch)         ALLOWED_BRANCH="$2"; shift 2 ;;
    -v|--version)     VERSION_ARG="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "error: unknown flag: $1" >&2; usage 2 ;;
    *)  if [[ -z "$VERSION_ARG" ]]; then VERSION_ARG="$1"; shift
        else echo "error: unexpected positional arg: $1" >&2; usage 2; fi ;;
  esac
done

if [[ -z "$VERSION_ARG" ]]; then
  echo "error: VERSION is required (e.g. 0.2.0)" >&2
  usage 2
fi

# Normalise: strip optional leading 'v'.
VERSION="${VERSION_ARG#v}"
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: version must be semver MAJOR.MINOR.PATCH (got '$VERSION_ARG')" >&2
  exit 2
fi
TAG="v${VERSION}"

# --- helpers -----------------------------------------------------------------

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
fail() { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '\033[2m   dry-run:\033[0m %s\n' "$*"
  else
    "$@"
  fi
}

# --- preflight ---------------------------------------------------------------

cd "$REPO_ROOT"

for tool in jq curl git python3; do
  command -v "$tool" >/dev/null 2>&1 || fail "missing required tool: $tool"
done

# Working tree clean?
if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "working tree is not clean — commit or stash changes first"
fi

# On the expected branch?
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$ALLOWED_BRANCH" ]]; then
  fail "expected branch '$ALLOWED_BRANCH' but on '$CURRENT_BRANCH' (use --branch to override)"
fi

# Tag must not already exist locally or on the remote.
if git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null; then
  fail "tag ${TAG} already exists locally"
fi
if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  fail "tag ${TAG} already exists on origin"
fi

# Origin must point at a gitlab.com project.
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$ORIGIN_URL" ]]; then
  fail "no 'origin' remote configured"
fi
PROJECT_PATH="$(printf '%s' "$ORIGIN_URL" \
  | sed -E 's#^https?://[^/]+/##; s#^git@[^:]+:##; s#\.git$##')"
if [[ -z "$PROJECT_PATH" ]] || [[ "$ORIGIN_URL" != *"gitlab.com"* ]]; then
  fail "origin does not look like a gitlab.com URL: $ORIGIN_URL"
fi
PROJECT_ENC="$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$PROJECT_PATH")"

log "repo:    $PROJECT_PATH"
log "tag:     $TAG"
log "branch:  $CURRENT_BRANCH"
[[ "$DRY_RUN" -eq 1 ]] && warn "DRY RUN — nothing will be changed"

# --- pull GitLab token -------------------------------------------------------

resolve_token() {
  if [[ -n "${GITLAB_TOKEN:-}" ]]; then
    printf '%s' "$GITLAB_TOKEN"
    return 0
  fi
  if [[ -r "${HOME}/.git-credentials" ]]; then
    # https://oauth2:TOKEN@gitlab.com  -or-  https://USER:TOKEN@gitlab.com
    awk -F '[/:@]' '/gitlab\.com/ {print $7; exit}' "${HOME}/.git-credentials" 2>/dev/null \
      || true
  fi
}
TOKEN="$(resolve_token || true)"
if [[ -z "$TOKEN" ]]; then
  fail "no GitLab token found — set GITLAB_TOKEN or store creds in ~/.git-credentials"
fi

# --- extract CHANGELOG section ----------------------------------------------

if [[ ! -f "$CHANGELOG" ]]; then
  fail "CHANGELOG.md not found at $CHANGELOG"
fi

# Pull the section starting at "## [VERSION] - DATE" up to the next "## [".
SECTION_BODY="$(awk -v ver="$VERSION" '
  $0 ~ "^## \\[" ver "\\]" { capture=1; next }
  capture && /^## \[/      { exit }
  capture                  { print }
' "$CHANGELOG")"

if [[ -z "$(printf '%s' "$SECTION_BODY" | tr -d '[:space:]')" ]]; then
  fail "no CHANGELOG section found for version $VERSION (expected '## [$VERSION] - YYYY-MM-DD')"
fi

# Trim leading/trailing blank lines.
SECTION_BODY="$(printf '%s\n' "$SECTION_BODY" | awk 'NF{found=1} found' | awk 'BEGIN{r=""} {r=r $0 "\n"} END{sub(/[[:space:]]+$/,"",r); print r}')"

RELEASE_TITLE="${TAG} — spark-benchmark"
TAG_MESSAGE="$(printf 'spark-benchmark %s\n\n%s\n' "$TAG" "$SECTION_BODY")"

log "extracted CHANGELOG section ($(printf '%s' "$SECTION_BODY" | wc -l | tr -d ' ') lines)"

# --- tag ---------------------------------------------------------------------

log "creating annotated tag $TAG"
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\033[2m   dry-run:\033[0m git tag -a %s -m "<changelog body>"\n' "$TAG"
else
  git tag -a "$TAG" -m "$TAG_MESSAGE"
fi

log "pushing tag to origin"
run git push origin "$TAG"

# --- GitLab Release ----------------------------------------------------------

log "creating GitLab Release"

PAYLOAD="$(jq -n \
  --arg tag_name "$TAG" \
  --arg name     "$RELEASE_TITLE" \
  --arg description "$SECTION_BODY" \
  '{tag_name:$tag_name, name:$name, description:$description}')"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\033[2m   dry-run:\033[0m POST https://gitlab.com/api/v4/projects/%s/releases\n' "$PROJECT_ENC"
  printf '\033[2m   payload preview:\033[0m\n'
  printf '%s\n' "$PAYLOAD" | jq '{tag_name, name, description: (.description | .[0:240] + "...")}'
  log "dry-run complete"
  exit 0
fi

HTTP_RESP="$(mktemp)"
trap 'rm -f "$HTTP_RESP"' EXIT
HTTP_CODE="$(curl -sS -o "$HTTP_RESP" -w '%{http_code}' \
  -X POST \
  --header "PRIVATE-TOKEN: $TOKEN" \
  --header 'Content-Type: application/json' \
  --data  "$PAYLOAD" \
  "https://gitlab.com/api/v4/projects/${PROJECT_ENC}/releases")"

if [[ "$HTTP_CODE" != "201" ]]; then
  warn "GitLab API returned HTTP $HTTP_CODE"
  cat "$HTTP_RESP" >&2
  fail "release creation failed — tag is already pushed; create the Release manually at https://gitlab.com/${PROJECT_PATH}/-/releases or rerun after fixing the cause"
fi

RELEASE_URL="$(jq -r '._links.self // empty' "$HTTP_RESP")"
log "release published: ${RELEASE_URL:-https://gitlab.com/${PROJECT_PATH}/-/releases/${TAG}}"
