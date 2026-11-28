# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RELEASE = Path(__file__).resolve().parent


class GitHubReleaseTest(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.repo = self.path / 'repo'
        self.remote = self.path / 'origin.git'
        self.bin = self.path / 'bin'
        self.bin.mkdir()
        self.env = {
            'PATH': f'{self.bin}:{Path(sys.executable).parent}:{os.environ["PATH"]}',
            'HOME': temp.name,
            'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_ALLOW_PROTOCOL': 'file',
        }
        self.run_command('git', 'init', '--bare', '--initial-branch=master', self.remote, cwd=self.path)
        self.run_command('git', 'init', '--initial-branch=master', self.repo, cwd=self.path)
        self.git('config', 'user.name', 'Test Author')
        self.git('config', 'user.email', 'author@example.test')
        self.git('remote', 'add', 'origin', self.remote)
        self.target = self.commit_file('version.txt', '1.2.3\n')
        self.git('switch', '-c', 'feature')
        self.side = self.commit_file('feature.txt', 'feature\n')
        self.git('switch', 'master')
        self.git('merge', '--no-ff', 'feature', '-m', 'Merge reviewed feature')
        self.dispatch = self.git('rev-parse', 'HEAD')
        self.future = self.commit_file('version.txt', '1.2.4\n')
        self.git('push', 'origin', 'master')
        self.git('switch', '--detach', self.dispatch)
        (self.repo / 'release').mkdir()
        (self.repo / 'release' / 'bump_patch_version.py').write_bytes((RELEASE / 'bump_patch_version.py').read_bytes())
        self.output = self.path / 'output'
        self.gh_log = self.path / 'gh-log'
        self.env.update(
            GITHUB_ACTOR='octocat',
            GITHUB_EVENT_NAME='workflow_dispatch',
            GITHUB_OUTPUT=str(self.output),
            GITHUB_REF='refs/heads/master',
            GITHUB_REPOSITORY='deepspeedai/DeepSpeed',
            GITHUB_SHA=self.dispatch,
            GITHUB_STEP_SUMMARY=str(self.path / 'summary'),
            RELEASE_VERSION='1.2.3',
            RUNNER_TEMP=temp.name,
            TARGET_SHA=self.target,
        )
        self.stub('curl', 'printf "%s" "${PYPI_STATUS:-404}"\n')
        self.stub(
            'gh', f'exec "{sys.executable}" - "$@" <<\'PY\'\n'
            'import json, os, subprocess, sys\n'
            'args = sys.argv[1:]\n'
            'if args[:2] == ["api", "graphql"]:\n'
            '    print(os.getenv("RELEASE_STATE", "missing"))\n'
            '    sys.exit(int(os.getenv("RELEASE_LOOKUP_FAILURE", "0")))\n'
            f'with open({str(self.gh_log)!r}, "a") as log:\n'
            '    print(json.dumps(args), file=log)\n'
            'if args[:2] == ["api", "--method"]:\n'
            '    fields = dict(arg.split("=", 1) for arg in args[5::2])\n'
            f'    subprocess.run(["git", "--git-dir", {str(self.remote)!r}, "update-ref",\n'
            '                    fields["ref"], fields["sha"], "0" * 40], check=True)\n'
            'elif args[:2] == ["release", "create"]:\n'
            '    sys.exit(int(os.getenv("RELEASE_FAILURE", "0")))\n'
            'elif args[:2] != ["pr", "create"]:\n'
            '    sys.exit("Unexpected gh command: " + repr(args))\n'
            'PY\n')

    def run_command(self, *args, cwd=None, env=None, check=True):
        return subprocess.run(args,
                              cwd=cwd or self.repo,
                              env={
                                  **self.env,
                                  **(env or {})
                              },
                              text=True,
                              capture_output=True,
                              check=check)

    def git(self, *args):
        return self.run_command('git', *args).stdout.strip()

    def commit_file(self, file_name, content):
        (self.repo / file_name).write_text(content)
        self.git('add', file_name)
        self.git('commit', '-m', f'Update {file_name}')
        return self.git('rev-parse', 'HEAD')

    def stub(self, command, script):
        path = self.bin / command
        path.write_text('#!/bin/sh\n' + script)
        path.chmod(0o755)

    def release(self, step, **env):
        return self.run_command('bash', RELEASE / 'github_release.sh', step, env=env, check=False)

    def gh_calls(self):
        if not self.gh_log.exists():
            return []
        return [json.loads(line) for line in self.gh_log.read_text().splitlines()]

    def assert_resolved(self, sha, **env):
        self.output.write_text('')
        result = self.release('resolve', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text().splitlines(), [f'sha={sha}', 'version=1.2.3'])

    def assert_rejected(self, step, **env):
        result = self.release(step, **env)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_dispatch_is_bounded_by_dispatch_time_master(self):
        self.assert_resolved(self.dispatch)
        self.assert_resolved(self.target, REQUESTED_SHA=self.target.upper())
        for bad in ('master', self.target[:12], 'f' * 40, self.side, self.future, self.target + '\n--other'):
            with self.subTest(value=bad):
                self.assert_rejected('resolve', REQUESTED_SHA=bad)
        self.assert_rejected('resolve', GITHUB_REF='refs/heads/feature')
        self.assert_rejected('resolve', GITHUB_REF='refs/tags/v1.2.3')
        self.assert_rejected('resolve', GITHUB_SHA=self.side)

    def test_tag_event_uses_tagged_commit_and_its_version(self):
        self.git('tag', '-a', 'v1.2.3', self.target, '-m', 'Legacy tag')
        tag_object = self.git('rev-parse', 'v1.2.3')
        self.assert_resolved(self.target,
                             GITHUB_EVENT_NAME='push',
                             GITHUB_REF='refs/tags/v1.2.3',
                             GITHUB_SHA=tag_object)
        self.assert_rejected('resolve', GITHUB_EVENT_NAME='push', GITHUB_REF='refs/tags/v1.2.4', GITHUB_SHA=tag_object)
        self.assert_rejected('resolve', GITHUB_EVENT_NAME='push', GITHUB_REF='refs/tags/v1.2.3', GITHUB_SHA=self.side)

    def test_dispatch_creates_tag_and_requests_generated_notes(self):
        result = self.release('release')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git('ls-remote', '--refs', 'origin', 'refs/tags/v1.2.3'),
                         f'{self.target}\trefs/tags/v1.2.3')
        _, release = self.gh_calls()
        self.assertEqual(release[:3], ['release', 'create', 'v1.2.3'])
        self.assertEqual(release[release.index('--target') + 1], self.target)
        self.assertIn('--verify-tag', release)
        self.assertIn('--generate-notes', release)

    def test_tag_push_reuses_annotated_tag_and_rejects_missing_or_moved_tag(self):
        self.assert_rejected('release', GITHUB_EVENT_NAME='push')
        self.assertEqual(self.gh_calls(), [])
        self.git('tag', '-a', 'v1.2.3', self.target, '-m', 'Legacy tag')
        self.git('push', 'origin', 'refs/tags/v1.2.3')
        result = self.release('release', GITHUB_EVENT_NAME='push')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[:2] for call in self.gh_calls()], [['release', 'create']])
        self.gh_log.unlink()
        result = self.release('release', GITHUB_EVENT_NAME='push', RELEASE_STATE='published')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[:2] for call in self.gh_calls()], [])
        self.assert_rejected('release', GITHUB_EVENT_NAME='push', TARGET_SHA=self.dispatch)
        self.assert_rejected('release', TARGET_SHA=self.dispatch)

    def test_release_errors_stop_before_upload(self):
        self.assert_rejected('release', PYPI_STATUS='200')
        self.assert_rejected('release', PYPI_STATUS='503')
        self.assert_rejected('release', TARGET_SHA=self.side)
        self.assert_rejected('release', RELEASE_STATE='published')
        self.assert_rejected('release', RELEASE_STATE='unpublished')
        self.assert_rejected('release', RELEASE_LOOKUP_FAILURE='1')
        self.assertEqual(self.gh_calls(), [])
        self.assert_rejected('release', RELEASE_FAILURE='1')

    def test_bump_skips_stale_master_and_existing_branch(self):
        self.assert_rejected('bump')
        self.assertEqual(self.gh_calls(), [])
        self.git('push', '--force', 'origin', f'{self.dispatch}:master')
        self.git('push', 'origin', f'{self.dispatch}:refs/heads/AutoPR/1.2.3')
        self.assert_rejected('bump')
        self.assertEqual(self.gh_calls(), [])
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/AutoPR/1.2.3'),
                         f'{self.dispatch}\trefs/heads/AutoPR/1.2.3')

    def test_bump_proposes_only_next_patch_on_fresh_master(self):
        self.git('switch', 'master')
        current = self.commit_file('version.txt', '1.2.3\n')
        self.git('push', 'origin', 'master')
        self.git('switch', '--detach', self.dispatch)
        result = self.release('bump')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.git('fetch', 'origin', 'refs/heads/AutoPR/1.2.3')
        self.assertEqual(self.git('rev-parse', 'FETCH_HEAD^'), current)
        self.assertEqual(self.git('diff', '--name-only', 'FETCH_HEAD^', 'FETCH_HEAD'), 'version.txt')
        self.assertEqual(self.git('show', 'FETCH_HEAD:version.txt'), '1.2.4')
        self.assertIn('Signed-off-by: octocat <octocat@users.noreply.github.com>',
                      self.git('log', '-1', '--format=%B', 'FETCH_HEAD'))
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/master'), f'{current}\trefs/heads/master')
        pr, = self.gh_calls()
        self.assertEqual(pr[:2], ['pr', 'create'])
        self.assertEqual(pr[pr.index('--base') + 1], 'master')
        self.assertEqual(pr[pr.index('--head') + 1], 'AutoPR/1.2.3')
        self.assertEqual(pr[pr.index('--assignee') + 1], 'octocat')

        self.git('switch', 'master')
        self.commit_file('version.txt', '1.2.9\n')
        merge = self.run_command('git', 'merge', '--no-commit', 'FETCH_HEAD', check=False)
        self.assertNotEqual(merge.returncode, 0)
        self.assertEqual(self.git('diff', '--name-only', '--diff-filter=U'), 'version.txt')
        self.assertEqual(self.git('show', 'HEAD:version.txt'), '1.2.9')


if __name__ == '__main__':
    unittest.main()
