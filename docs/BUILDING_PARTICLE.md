# Build Particle from source

Requires an Apple Silicon Mac, macOS 26+, Swift 6.2+, and Xcode with its Metal tools. Model weights are installed separately through the app's pinned public catalog. No Hugging Face token is required for that catalog.

For the original development layout, Xcode and all compiler/dependency caches live on the Particle SD volume:

```sh
Scripts/particle doctor
Scripts/particle build -c release --product ParticleOne
Scripts/particle build -c release --product ParticleCLI
Scripts/particle test -c release --filter 'AgentHarnessTests|AgentPromptRenderingTests|ParticleOneTests|MessageEditingTests|NoThinkingTests|ThinkingControlTests|WorkspacePolishTests'
Scripts/build_particleone.sh --package-only --version 0.3.1
```

For another development disk, set `PARTICLE_SD_ROOT` to its mounted path and place Xcode.app there. `PARTICLE_TOOLCHAIN=clt` explicitly selects Command Line Tools while retaining build/cache placement on the selected volume. See Scripts/particle-env.sh for paths. An installed application bundles its runtime/shaders and does not require Xcode or the SD card. User data stays in the compatibility directory `~/Library/Application Support/ParticleOne`.

The package retains ParticleOne target and executable names and the original bundle identifier for data/update compatibility; its visible name is Particle. The `.particle` model is a directory pack rather than a single flat file.

Packaging produces an ad-hoc-signed local app/archive and a local Homebrew recipe. Production distribution requires your own update-signing key, HTTPS update/archive hosting, code signing and notarization. Never reuse another publisher's private key. Scripts/particleone/update-channel.json contains only a public verification key and the original loopback development feed. Local publishing requires the matching key in Keychain. See [updates](particleone/UPDATES.md).

Git ignores local data, model weights, build products, credentials, developer settings and machine-specific validation logs. Preserve LICENSE, THIRD_PARTY_NOTICES.md, bundled font licenses and model notices when redistributing source or binaries.
