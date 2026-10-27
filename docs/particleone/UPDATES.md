# In-app updates

ParticleOne uses Sparkle 2.9.6. The application checks its signed appcast at launch. A newer compatible build shows an arrow-in-circle icon at the top right, without restoring the removed top bar. Settings → App updates and ParticleOne → Check for updates provide manual controls. The icon opens Sparkle’s download/install interface; installation requires the user to choose it. Updates verify Ed25519 signatures before extraction and replace the same application, then restart it. Conversations, settings, memories and models are outside the bundle and are not included in update archives.

Updates cannot start during loading or generation. If a response begins while the installer dialog is open, restart preparation waits for that response to finish, saves state and shuts down the single owned model worker before proceeding.

## One-time bootstrap on this Mac

The older app cannot gain an updater without being replaced once. Run `Scripts/particleone/install_bootstrap.py` after building the first updater-enabled release. This installs `~/Applications/Particle.app` without launching it or replacing a running app. Quit any older copy normally, then open this stable copy. Subsequent updates replace this location through Sparkle.

## Publish future local builds

```sh
cd /Users/lakshyaprajapati/Repositories/Particle
Scripts/release_particleone.sh
```

This compiles and packages into a reusable internal staging directory, assigns an increasing UTC timestamp build number, verifies the archive and code signature, signs both archive and appcast, and publishes the feed. No version-named app folder needs to be opened. Use `--version 0.3.1` to change the display version; build numbers still increase automatically. Keep the previous release archives available for existing downloads.

The development feed is `http://127.0.0.1:49271/appcast.xml`. A per-user LaunchAgent named `app.particleone.update-host` serves only the signed feed and app archives, bound exclusively to loopback. Directory listings, symlinks and other files are rejected. Host and release files live under `~/Library/Application Support/ParticleOneUpdates`, separate from workspace data. The local app polls once per minute; a build reaches an open updater-enabled copy on its next check. Settings also permits an immediate check. Failed offline checks retain the installed app.

The private signing key stays in macOS Keychain under account `particleone-publisher`; only its public key is in `update-channel.json` and the app. Losing this key requires a separately planned trust migration. Never put it into source or an app bundle. Publishing does not silently install or restart the user's app.

To stop the development server: `launchctl bootout gui/$(id -u)/app.particleone.update-host`. Its plist is `~/Library/LaunchAgents/app.particleone.update-host.plist`. Stopping it does not remove the app or model. This is a local development channel, not an Internet push service.

## Public distribution

No public app archive/feed has been published. Supply an HTTPS `--update-feed` and matching `--update-public-key` when packaging a public build, host a signed appcast and signed archives, and use a Developer ID identity plus notarization for distribution. HTTPS builds poll every six hours. The local publisher intentionally refuses nonlocal channels. Model releases remain independent of app updates. The app supports both Homebrew cask installation and in-app updates; do not run both installers concurrently.

References: [Sparkle setup](https://sparkle-project.org/documentation/), [publishing](https://sparkle-project.org/documentation/publishing/), [customization](https://sparkle-project.org/documentation/customization/).

From 0.3.0 the product/bundle is Particle.app and archives use the Particle- prefix. Existing bundle identity, update key, localhost feed and compatibility cask token remain unchanged. The server accepts both old and new archive names.
