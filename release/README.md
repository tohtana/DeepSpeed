# Release DeepSpeed

## Request a release

Anyone with write access to `deepspeedai/DeepSpeed` can request a release.

1. In the upstream repository's **Actions** tab, open [Build and publish DeepSpeed release](https://github.com/deepspeedai/DeepSpeed/actions/workflows/release.yml).
2. Click **Run workflow** and select `master`. Leave the SHA field blank to release the commit at the tip of `master` when you start the run. To select an earlier reviewed commit, enter its full 40-character SHA from the first-parent history of `master`.
3. Click **Run workflow** to submit the request. Open the new run. The **Verify release target** summary shows the selected SHA and the version from that commit's `version.txt`.

## Approve and publish

One of the required reviewers for `release-env` approves publication in the same run.

1. Check that the SHA and version in **Verify release target** match the intended release.
2. At the deployment review prompt, click **Review deployments**, select **release-env**, then click **Approve and deploy**.

The workflow builds the source distribution from the approved SHA, even if `master` has advanced. It creates or verifies the matching version tag, creates the GitHub Release with generated notes, then uploads the package to [PyPI](https://pypi.org/project/deepspeed/).

## Merge the next version bump

After the PyPI upload succeeds, the same run opens a PR against `master`. Review and merge it through the usual checks. For example, publishing `0.20.1` creates a PR to change `version.txt` to `0.20.2`. Merging that PR prepares the next development version.

## If the run fails

Check the failed step, the tag's SHA, the [GitHub Release](https://github.com/deepspeedai/DeepSpeed/releases), and the version on PyPI before retrying. A manual run stops if the GitHub Release or PyPI version already exists.

If PyPI already has the version and only the PR step failed, leave the published release in place. Check `version.txt` on `master` and look for an existing `AutoPR/<released-version>` branch or PR. If the bump is still needed, use the existing PR or open one for the patch increment.

## Release from a tag

Pushing a new `vX.Y.Z` tag on a commit containing this workflow also starts a release. The tag must point to a first-parent commit on `master` and match its `version.txt`. Approve the run in `release-env` as above. If publishing a stable GitHub Release created the tag, the workflow keeps its release notes.
