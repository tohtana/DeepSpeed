# Release DeepSpeed

1. In the upstream repository, open **Actions → Build and publish DeepSpeed release → Run workflow**. Select `master`.
2. Enter the full SHA of the reviewed commit on the first-parent history of `master`. Leave the field blank to use the commit at dispatch time.
3. Check the SHA in the run name and the version in the **Verify release target** job summary before approving `release-env`.
4. After the run publishes the GitHub Release and PyPI source distribution, review and merge the version bump PR through the usual checks.

A push of a `vX.Y.Z` tag on a commit with this workflow also works. The tag must point to a first-parent commit on `master` whose `version.txt` matches the tag. If publishing a GitHub Release created the tag, the workflow keeps its release notes and continues with PyPI publication.

If the run fails, check the tag, GitHub Release, and PyPI before retrying. An existing PyPI version stops publication. An existing GitHub Release also stops a manual dispatch. If PyPI succeeded and the bump PR failed, check the current `version.txt` on `master` and open the bump PR manually if needed.
