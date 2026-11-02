# Install Particle for testing

Requires an Apple Silicon Mac and macOS 26 or newer. Xcode, Git, Homebrew,
Python and a GitHub account are not required. The installer uses macOS tools.

Paste into Terminal:

```bash
curl -fsSL https://github.com/MathObsession/Particle/releases/latest/download/install.sh | /bin/bash
```

This downloads the prebuilt application from a versioned GitHub release,
checks its pinned SHA-256, bundle identifier, version, build and code signature,
installs it at `~/Applications/Particle.app` and opens it. On first launch the
app downloads the current Churo model (approximately 19.6 GB); allow at least
22 GB free for the model plus room for the app. Chronos support is not included.

The tester build has an ad-hoc development signature and is not Apple-notarized.
If macOS blocks it, use System Settings → Privacy & Security → Open Anyway
after attempting to open Particle. The installer does not change Gatekeeper,
disable quarantine or request administrator access. Apple may require further
confirmation; this is not a promise of unattended installation.

Quit Particle and its CLI before rerunning the installer. After successful
replacement, the old app copy at that destination and a verified legacy
`~/Applications/ParticleOne.app` are removed. A newer installed build or an
unrelated application is never replaced. Models, chats, settings and memories
are preserved. App copies elsewhere are not searched for or deleted by this
public installer.

The existing app's update feed still targets the publisher's local development
server. Public in-app automatic updates are not configured in this release;
use the release installer for subsequent public tester builds.

To inspect the installer before running it, download `install.sh` from
[GitHub Releases](https://github.com/MathObsession/Particle/releases/latest).
Running it with `--verify-only` downloads and verifies the archive without
installing or launching it:

```bash
/bin/bash install.sh --verify-only
```

Maintainers: `Scripts/particleone/install.sh` pins the archive URL, version,
build and checksum. Publish that script with the exact matching prebuilt ZIP
in a release. Do not reuse the checksum for a different archive. The existing
0.3.3 binary corresponds to source commit
`6249a805fc011d03d34c75f6be9560e221ed5a9b`.

Installer regression checks use the real archive, isolated temporary app/data
folders and a substituted download/launch step; no model runs or user apps are
started:

```bash
PARTICLE_TEST_ARCHIVE=/absolute/path/to/Particle-0.3.3-20261027142620-arm64.zip \
  python3 Scripts/particleone/test_tester_installer.py
```
