#!/usr/bin/env bash
# tools/sync-upstream.sh - pull upstream Strata (Niko1221/Strata) into Xolotl without losing Xolotl's own work.
#
#   tools/sync-upstream.sh            fetch, merge into a fresh branch up/merge-<date> in its own worktree, report
#   tools/sync-upstream.sh --build    ... and build the engines there (build-merge/) with build-xolotl's options
#
# What makes repeat syncs cheap:
#   * a real `git merge` (never a rebase / cherry-picks): every Xolotl commit stays as it is
#   * git rerere (enabled here): a conflict resolved once is resolved the same way automatically next time
#   * the lost-lines audit: every line Xolotl added since the last merge base must still be in the merged tree;
#     anything else is listed (a rewrite that keeps the behaviour is fine, a dropped feature is not)
#   * docs/FORK_PATCHES.md: where Xolotl patches upstream files and how each conflict was settled last time
# Merge often (every week or two): a few dozen commits conflict in a handful of places, 2,500 conflict everywhere.
#
# The branch is only a candidate: `public` moves only after the build and the regression runs pass
# (docs/FORK_PATCHES.md, "Accepting a merge").
set -euo pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$REPO"
REMOTE=${STRATA_UPSTREAM_REMOTE:-niko}
git remote get-url "$REMOTE" >/dev/null 2>&1 || git remote add "$REMOTE" https://github.com/Niko1221/Strata.git
git config rerere.enabled true
git config rerere.autoupdate true
git fetch "$REMOTE" --quiet
BASE=public
NEW=$(git rev-list --count "$BASE".."$REMOTE"/main)
echo "upstream: $NEW commits not in $BASE (merge base $(git merge-base "$BASE" "$REMOTE"/main | cut -c1-10))"
[ "$NEW" = 0 ] && { echo "nothing to merge"; exit 0; }

DATE=$(date +%Y-%m-%d)
BR=up/merge-$DATE
WT="$REPO/../Xolotl-merge-$DATE"
git worktree add -b "$BR" "$WT" "$BASE" >/dev/null
cd "$WT"
git config rerere.enabled true
git config rerere.autoupdate true
if git merge --no-ff --no-edit "$REMOTE"/main >/dev/null 2>&1; then
    echo "merged cleanly (rerere replayed any known resolutions)"
else
    git rerere >/dev/null 2>&1 || true
    LEFT=$(git diff --name-only --diff-filter=U)
    echo "conflicts left for a person ($(echo "$LEFT" | grep -c . || true) files) - see docs/FORK_PATCHES.md for how each was settled before:"
    echo "$LEFT" | sed 's/^/  /'
    echo "resolve in $WT, then: git add -A && git commit --no-edit, and run this script's audit step:"
    echo "  python3 tools/lost_lines.py \$(git merge-base $BASE $REMOTE/main) $BASE HEAD"
    exit 2
fi
python3 tools/lost_lines.py "$(git merge-base "$BASE" "$REMOTE"/main)" "$BASE" HEAD | head -60
if [ "${1:-}" = "--build" ]; then
    OPTS=$(grep -E "^STRATA_[A-Z0-9_]*:(BOOL|STRING|PATH)=" "$REPO/build-xolotl/CMakeCache.txt" | grep -vE "=$|=OFF$" |
           sed -E 's/^([A-Z0-9_]+):[A-Z]+=(.*)$/-D\1=\2/' | tr '\n' ' ')
    cmake -S . -B build-merge -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 $OPTS >/dev/null
    nice -n 10 ninja -C build-merge strata ds41_generate glm_generate strata-dense
fi
echo "candidate: branch $BR in $WT"
