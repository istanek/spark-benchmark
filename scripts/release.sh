#!/usr/bin/env bash
# scripts/release.sh — cut a spark-benchmark release
#
# What it does:
#   1. Validates the working tree is clean and we're on main (override with --branch).
#   2. Reads the matching `## [X.Y.Z] - YYYY-MM-DD` section out of CHANGELOG.md.
#   3. Creates an annotated tag `vX.Y.Z` whose message is that section.
#   4. Pushes the tag to origin.
#   5. Creates a GitHub Release whose body is the same section.
#
# Usage:
#   scripts/release.sh 0.2.0                 # cut v0.2.0
#   scripts/release.sh v0.2.0                # same, leading v is allowed
#   scripts/release.sh 0.2.0 --dry-run       # show what would happen, change nothing
#   scripts/release.sh 0.2.0 --branch foo    # allow cutting from a non-main branch
#
# Requirements:
#   - jq, curl, git, python3
#   - GITHUB_TOKEN env var, or `gh auth token` working, or a token stored in
#     ~/.git-credentials for github.com. The token must have the `repo` scope
#     (or, on a fine-grained PAT, `Contents: read & write`) to create a Release.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly CHANGELOG="${REPO_ROOT}/CHANGELOG.md"

# --- args --------------------------------------------------------------------

DRY_RUN=0
ALLOWED_BRANCH="main"
VERSION_ARG=""

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
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

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "working tree is not clean — commit or stash changes first"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$ALLOWED_BRANCH" ]]; then
  fail "expected branch '$ALLOWED_BRANCH' but on '$CURRENT_BRANCH' (use --branch to override)"
fi

if git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null; then
  fail "tag ${TAG} already exists locally"
fi
if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  fail "tag ${TAG} already exists on origin"
fi

# Origin must point at a github.com project.
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$ORIGIN_URL" ]]; then
  fail "no 'origin' remote configured"
fi
PROJECT_PATH="$(printf '%s' "$ORIGIN_URL" \
  | sed -E 's#^https?://[^/]+/##; s#^git@[^:]+:##; s#\.git$##')"
if [[ -z "$PROJECT_PATH" ]] || [[ "$ORIGIN_URL" != *"github.com"* ]]; then
  fail "origin does not look like a github.com URL: $ORIGIN_URL"
fi
OWNER="${PROJECT_PATH%%/*}"
REPO="${PROJECT_PATH#*/}"
if [[ -z "$OWNER" ]] || [[ -z "$REPO" ]] || [[ "$OWNER" == "$PROJECT_PATH" ]]; then
  fail "could not parse owner/repo from origin URL: $ORIGIN_URL"
fi

log "repo:    $PROJECT_PATH"
log "tag:     $TAG"
log "branch:  $CURRENT_BRANCH"
[[ "$DRY_RUN" -eq 1 ]] && warn "DRY RUN — nothing will be changed"

# --- pull GitHub token -------------------------------------------------------

resolve_token() {
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    printf '%s' "$GITHUB_TOKEN"
    return 0
  fi
  if command -v gh >/dev/null 2>&1; then
    local gh_tok
    gh_tok="$(gh auth token 2>/dev/null || true)"
    if [[ -n "$gh_tok" ]]; then
      printf '%s' "$gh_tok"
      return 0
    fi
  fi
  if [[ -r "${HOME}/.git-credentials" ]]; then
    # https://oauth2:TOKEN@github.com  -or-  https://USER:TOKEN@github.com
    awk -F '[/:@]' '/github\.com/ {print $7; exit}' "${HOME}/.git-credentials" 2>/dev/null \
      || true
  fi
}
TOKEN="$(resolve_token || true)"
if [[ -z "$TOKEN" ]]; then
  fail "no GitHub token found — set GITHUB_TOKEN, run \`gh auth login\`, or store creds in ~/.git-credentials"
fi

# --- extract CHANGELOG section ----------------------------------------------

if [[ ! -f "$CHANGELOG" ]]; then
  fail "CHANGELOG.md not found at $CHANGELOG"
fi

SECTION_BODY="$(awk -v ver="$VERSION" '
  $0 ~ "^## \\[" ver "\\]" { capture=1; next }
  capture && /^## \[/      { exit }
  capture                  { print }
' "$CHANGELOG")"

if [[ -z "$(printf '%s' "$SECTION_BODY" | tr -d '[:space:]')" ]]; then
  fail "no CHANGELOG section found for version $VERSION (expected '## [$VERSION] - YYYY-MM-DD')"
fi

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

# --- GitHub Release ----------------------------------------------------------

log "creating GitHub Release"

PAYLOAD="$(jq -n \
  --arg tag_name "$TAG" \
  --arg name     "$RELEASE_TITLE" \
  --arg body     "$SECTION_BODY" \
  '{tag_name:$tag_name, name:$name, body:$body, draft:false, prerelease:false}')"

RELEASE_URL_API="https://api.github.com/repos/${OWNER}/${REPO}/releases"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\033[2m   dry-run:\033[0m POST %s\n' "$RELEASE_URL_API"
  printf '\033[2m   payload preview:\033[0m\n'
  printf '%s\n' "$PAYLOAD" | jq '{tag_name, name, draft, prerelease, body: (.body | .[0:240] + "...")}'
  log "dry-run complete"
  exit 0
fi

HTTP_RESP="$(mktemp)"
trap 'rm -f "$HTTP_RESP"' EXIT
HTTP_CODE="$(curl -sS -o "$HTTP_RESP" -w '%{http_code}' \
  -X POST \
  --header "Authorization: Bearer $TOKEN" \
  --header "Accept: application/vnd.github+json" \
  --header "X-GitHub-Api-Version: 2022-11-28" \
  --header 'Content-Type: application/json' \
  --data  "$PAYLOAD" \
  "$RELEASE_URL_API")"

if [[ "$HTTP_CODE" != "201" ]]; then
  warn "GitHub API returned HTTP $HTTP_CODE"
  cat "$HTTP_RESP" >&2
  fail "release creation failed — tag is already pushed; create the Release manually at https://github.com/${OWNER}/${REPO}/releases/new?tag=${TAG} or rerun after fixing the cause"
fi

RELEASE_HTML_URL="$(jq -r '.html_url // empty' "$HTTP_RESP")"
log "release published: ${RELEASE_HTML_URL:-https://github.com/${OWNER}/${REPO}/releases/tag/${TAG}}"
