#!/usr/bin/env bash
# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
    echo "::error::$*" >&2
    exit 1
}

require_master_commit() {
    git rev-list --first-parent "$2" | grep -Fx "$1" >/dev/null ||
        die "Commit $1 is outside the required first-parent history of master."
}

resolve_target() {
    local target version event_commit
    event_commit=$(git rev-parse --verify "${GITHUB_SHA}^{commit}")
    case "$GITHUB_EVENT_NAME" in
        workflow_dispatch)
            [[ "$GITHUB_REF" == refs/heads/master ]] || die "Run this workflow from master."
            target=${REQUESTED_SHA:-$event_commit}
            [[ "$target" =~ ^[0-9a-fA-F]{40}$ ]] || die "Enter a full 40-character commit SHA."
            target=$(printf '%s' "$target" | tr 'A-F' 'a-f')
            require_master_commit "$event_commit" origin/master
            require_master_commit "$target" "$event_commit"
            ;;
        push)
            target=$event_commit
            require_master_commit "$target" origin/master
            ;;
        *)
            die "Unsupported release event."
            ;;
    esac

    version=$(git show "$target:version.txt")
    [[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] ||
        die "The target version.txt must contain a version in X.Y.Z form."
    if [[ "$GITHUB_EVENT_NAME" == push && "$GITHUB_REF" != "refs/tags/v$version" ]]; then
        die "The pushed tag must match version.txt at the tagged commit."
    fi
    printf 'sha=%s\nversion=%s\n' "$target" "$version" >> "$GITHUB_OUTPUT"
    printf 'Release v%s from commit %s.\n' "$version" "$target" >> "$GITHUB_STEP_SUMMARY"
}

build_sdist() {
    local source_dir="$RUNNER_TEMP/release-source"
    local venv="$RUNNER_TEMP/release-venv"
    git worktree add --detach "$source_dir" "${TARGET_SHA}^{commit}"
    [[ "$(git -C "$source_dir" rev-parse HEAD)" == "$TARGET_SHA" ]] || die "Source commit changed."

    python -m venv "$venv"
    "$venv/bin/python" -m pip install build packaging
    (
        cd "$source_dir"
        DS_BUILD_OPS=0 DS_BUILD_STRING=" " "$venv/bin/python" -m build --sdist --outdir "$GITHUB_WORKSPACE/dist"
    )
    shopt -s nullglob
    local archives=("$GITHUB_WORKSPACE"/dist/*)
    [[ "${#archives[@]}" == 1 && "${archives[0]}" == "$GITHUB_WORKSPACE/dist/deepspeed-$RELEASE_VERSION.tar.gz" ]] ||
        die "Expected one source distribution for DeepSpeed $RELEASE_VERSION."
    printf '%s\n' "$venv/bin" >> "$GITHUB_PATH"
}

remote_tag_commit() {
    local ref="refs/tags/$1"
    git ls-remote origin "$ref" "$ref^{}" |
        awk -v ref="$ref" '$2 == ref { tag = $1 } $2 == ref "^{}" { peeled = $1 } END { print (peeled ? peeled : tag) }'
}

create_release() {
    local tag="v$RELEASE_VERSION"
    local tagged_commit status release_state
    git fetch --no-tags origin refs/heads/master
    require_master_commit "$TARGET_SHA" FETCH_HEAD

    status=$(curl --silent --show-error --max-time 30 --output /dev/null --write-out '%{http_code}' \
        "https://pypi.org/pypi/deepspeed/$RELEASE_VERSION/json")
    case "$status" in
        404) ;;
        200) die "DeepSpeed $RELEASE_VERSION already exists on PyPI." ;;
        *) die "Could not check PyPI for DeepSpeed $RELEASE_VERSION (HTTP $status)." ;;
    esac

    tagged_commit=$(remote_tag_commit "$tag")
    if [[ -n "$tagged_commit" && "$tagged_commit" != "$TARGET_SHA" ]]; then
        die "Tag $tag points to a different commit."
    fi

    release_state=$(
        gh api graphql -F query=@- \
            -f owner="${GITHUB_REPOSITORY%/*}" -f name="${GITHUB_REPOSITORY#*/}" -f tag="$tag" \
            --jq '.data.repository.release | if . == null then "missing" elif .isDraft or .isPrerelease then "unpublished" else "published" end' <<'GRAPHQL'
query($owner: String!, $name: String!, $tag: String!) {
  repository(owner: $owner, name: $name) {
    release(tagName: $tag) { isDraft isPrerelease }
  }
}
GRAPHQL
    )
    case "$release_state" in
        missing) ;;
        published)
            [[ "$GITHUB_EVENT_NAME" == push ]] || die "GitHub Release $tag already exists."
            ;;
        *) die "GitHub Release $tag is a draft or prerelease." ;;
    esac

    if [[ -z "$tagged_commit" && "$GITHUB_EVENT_NAME" == workflow_dispatch ]]; then
        gh api --method POST "repos/$GITHUB_REPOSITORY/git/refs" \
            -f ref="refs/tags/$tag" -f sha="$TARGET_SHA" >/dev/null
        tagged_commit=$(remote_tag_commit "$tag")
    fi
    [[ "$tagged_commit" == "$TARGET_SHA" ]] || die "Tag $tag is missing or points to a different commit."

    if [[ "$release_state" == missing ]]; then
        gh release create "$tag" --repo "$GITHUB_REPOSITORY" --target "$TARGET_SHA" --verify-tag --generate-notes
    fi
    [[ "$(remote_tag_commit "$tag")" == "$TARGET_SHA" ]] || die "Tag $tag changed during release preparation."

}

open_bump_pr() {
    local bump_script="$PWD/release/bump_patch_version.py"
    local bump_dir="$RUNNER_TEMP/release-bump"
    local branch="AutoPR/$RELEASE_VERSION"
    local email="$GITHUB_ACTOR@users.noreply.github.com"
    local master_version
    git fetch --no-tags origin refs/heads/master
    master_version=$(git show FETCH_HEAD:version.txt)
    [[ "$master_version" == "$RELEASE_VERSION" ]] ||
        die "master now has version $master_version. Check whether a manual bump PR is needed."

    git worktree add --detach "$bump_dir" FETCH_HEAD
    cd "$bump_dir"
    python "$bump_script" --current_version "$RELEASE_VERSION"
    git add -- version.txt
    git -c user.name="$GITHUB_ACTOR" -c user.email="$email" commit --signoff -m "Bump version after $RELEASE_VERSION release"
    # An empty lease requires a new branch, preserving any PR already under review.
    git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push \
        --force-with-lease="refs/heads/$branch:" origin "HEAD:refs/heads/$branch"
    gh pr create --repo "$GITHUB_REPOSITORY" --base master --head "$branch" \
        --assignee "$GITHUB_ACTOR" \
        --title "Update version.txt after $RELEASE_VERSION release" \
        --body "Bump the patch version after the DeepSpeed $RELEASE_VERSION release."
}

case "$1" in
    resolve) resolve_target ;;
    build) build_sdist ;;
    release) create_release ;;
    bump) open_bump_pr ;;
    *) die "Unknown release step." ;;
esac
