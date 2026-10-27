#!/usr/bin/env bash
set -euo pipefail
particle_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$particle_workspace/Scripts/particle-env.sh"
if [[ "${1:-}" != "--package-only" ]]; then
  "$particle_workspace/Scripts/particle" build -c release --product ParticleOne
  "$particle_workspace/Scripts/particle" build -c release --product ParticleCLI
else
  shift
fi
exec python3 "$particle_workspace/Scripts/particleone/package_app.py" --bin "$PARTICLE_BUILD_ROOT/SwiftPM/release" "$@"
