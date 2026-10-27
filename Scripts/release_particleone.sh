#!/usr/bin/env bash
# Future releases update the same installed app through the signed feed.
set -euo pipefail
particle_release_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$particle_release_workspace/Scripts/build_particleone.sh" --output "$particle_release_workspace/dist/update-staging" "$@"
python3 "$particle_release_workspace/Scripts/particleone/publish_update.py"
