#!/usr/bin/env python3
"""Exercise the public installer against the real release without network or app launch."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = Path(os.environ['PARTICLE_TEST_ARCHIVE']).resolve()
        cls.original = Path(__file__).with_name('install.sh').read_text()

    def run_installer(self, directory, *, verify=False, corrupt=False):
        root = Path(directory)
        home = root / 'home'
        home.mkdir(exist_ok=True)
        downloader = root / 'download'
        downloader.write_text('''#!/bin/bash
set -eu
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --output ]]; then
    shift
    if [[ "${PARTICLE_TEST_CORRUPT:-}" == 1 ]]; then
      printf 'invalid archive' > "$1"
    else
      /bin/cp "$PARTICLE_TEST_ARCHIVE" "$1"
    fi
    exit 0
  fi
  shift
done
exit 2
''')
        downloader.chmod(0o700)
        script = root / 'install.sh'
        script.write_text(self.original.replace('/usr/bin/curl ', f'"{downloader}" ').replace(
            '/bin/ps -axww -o comm=', '/usr/bin/printf ""').replace(
            '/usr/bin/open "$target"', '/usr/bin/printf "launch requested\\n"'))
        env = dict(os.environ, HOME=str(home), PARTICLE_TEST_ARCHIVE=str(self.archive),
                   PARTICLE_TEST_CORRUPT='1' if corrupt else '')
        args = ['/bin/bash', str(script)] + (['--verify-only'] if verify else [])
        return subprocess.run(args, env=env, text=True, capture_output=True)

    def test_verified_archive_does_not_install(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_installer(directory, verify=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('No app was installed', result.stdout)
            self.assertFalse((Path(directory) / 'home/Applications').exists())

    def test_bad_checksum_leaves_destination_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_installer(directory, corrupt=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum did not match', result.stderr)
            self.assertFalse((Path(directory) / 'home/Applications').exists())

    def test_install_and_replace_preserve_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / 'home/Library/Application Support/ParticleOne'
            support.mkdir(parents=True)
            (support / 'private-fixture').write_text('keep chats and models')
            for _ in range(2):
                result = self.run_installer(directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('launch requested', result.stdout)
                self.assertTrue((root / 'home/Applications/Particle.app').is_dir())
                self.assertEqual(list((root / 'home/Applications').glob('.particle-install.*')), [])
                self.assertEqual((support / 'private-fixture').read_text(), 'keep chats and models')

    def test_unrelated_application_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'home/Applications/Particle.app'
            target.mkdir(parents=True)
            marker = target / 'unrelated'
            marker.write_text('do not replace')
            result = self.run_installer(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), 'do not replace')


if __name__ == '__main__':
    unittest.main()
