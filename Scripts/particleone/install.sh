#!/bin/bash
# Particle's public tester installer. Uses only tools included with macOS.
set -euo pipefail

version='0.5.0'
build='20261104161646'
archive='Particle-0.5.0-20261104161646-arm64.zip'
sha256='b152adf62d75068bfa4314d2837239d3db5a0c31cb7f6c6c62fb870cda9a18a8'
base='https://github.com/MathObsession/Particle/releases/download/v0.5.0'
identity='app.particleone.desktop'
verify_only=false
case "${1:-}" in
  '') ;;
  --verify-only) verify_only=true ;;
  *) printf 'Usage: install.sh [--verify-only]\n' >&2; exit 2 ;;
esac
fail() { printf 'Particle: %s\n' "$*" >&2; exit 1; }
[[ "$(/usr/bin/uname -s)" == Darwin ]] || fail 'An Apple Silicon Mac is required.'
[[ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" == 1 ]] || fail 'An Apple Silicon Mac is required.'
os_version="$(/usr/bin/sw_vers -productVersion)"
[[ "${os_version%%.*}" -ge 26 ]] || fail 'macOS 26 or newer is required.'

temporary="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/particle-install.XXXXXX")"
stage=''
backup=''
target=''
installed=false
cleanup() {
  if [[ "$installed" == false && -n "$backup" && -d "$backup" && ! -e "$target" ]]; then
    /bin/mv "$backup" "$target"
  fi
  [[ -z "$stage" ]] || /bin/rm -rf "$stage"
  /bin/rm -rf "$temporary"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Downloading Particle %s…\n' "$version"
/usr/bin/curl --fail --location --proto '=https' --proto-redir '=https' \
  --retry 3 --connect-timeout 30 --output "$temporary/$archive" "$base/$archive"
actual="$(/usr/bin/shasum -a 256 "$temporary/$archive")"
[[ "${actual%% *}" == "$sha256" ]] || fail 'The download checksum did not match. Nothing was installed.'
/usr/bin/ditto -x -k "$temporary/$archive" "$temporary/unpacked"
source_app="$temporary/unpacked/Particle.app"
plist="$source_app/Contents/Info.plist"
[[ -d "$source_app" && ! -L "$source_app" ]] || fail 'The archive does not contain Particle.app.'
read_plist() { /usr/libexec/PlistBuddy -c "Print :$2" "$1/Contents/Info.plist"; }
[[ "$(read_plist "$source_app" CFBundleIdentifier)" == "$identity" ]] || fail 'Unexpected application identity.'
[[ "$(read_plist "$source_app" CFBundleShortVersionString)" == "$version" ]] || fail 'Unexpected application version.'
[[ "$(read_plist "$source_app" CFBundleVersion)" == "$build" ]] || fail 'Unexpected application build.'
/usr/bin/codesign --verify --deep --strict "$source_app"
if "$verify_only"; then
  printf 'Particle %s verified: checksum, identity, build and code signature. No app was installed.\n' "$version"
  exit 0
fi

apps="$HOME/Applications"
target="$apps/Particle.app"
legacy="$apps/ParticleOne.app"
[[ ! -L "$apps" ]] || fail '~/Applications is a symbolic link; install manually instead.'
/bin/mkdir -p "$apps"
for candidate in "$target" "$legacy"; do
  [[ ! -L "$candidate" ]] || fail 'An application destination is a symbolic link; install manually instead.'
  if [[ -e "$candidate" ]]; then
    [[ "$(read_plist "$candidate" CFBundleIdentifier)" == "$identity" ]] || fail 'An unrelated app occupies the destination.'
    /usr/bin/codesign --verify --deep --strict "$candidate"
    existing_build="$(read_plist "$candidate" CFBundleVersion)"
    [[ "$existing_build" =~ ^[0-9]+$ ]] || fail 'The installed build number is not recognized.'
    [[ "$existing_build" -le "$build" ]] || fail 'A newer Particle build is already installed.'
  fi
done
processes="$(/bin/ps -axww -o comm=)"
if [[ "$processes" == *'/Particle.app/Contents/'* || "$processes" == *'/ParticleOne.app/Contents/'* || "$processes" == *'/ParticleCLI'* || "$processes" == *'/ParticleMLXCLI'* || "$processes" == *'/ParticleLibraryCLI'* ]]; then
  fail 'Quit Particle and any Particle CLI session, then run this installer again. No process was stopped.'
fi
stage="$(/usr/bin/mktemp -d "$apps/.particle-install.XXXXXX")"
/usr/bin/ditto "$source_app" "$stage/Particle.app"
/usr/bin/codesign --verify --deep --strict "$stage/Particle.app"
if [[ -e "$target" ]]; then
  backup="$stage/previous.app"
  /bin/mv "$target" "$backup"
fi
/bin/mv "$stage/Particle.app" "$target"
installed=true
# Only the replaced, verified application copies are removed. User data is separate.
[[ -z "$backup" ]] || /bin/rm -rf "$backup"
[[ ! -e "$legacy" ]] || /bin/rm -rf "$legacy"
printf 'Installed: %s\n' "$target"
printf 'The app downloads its model on first launch (about 19.6 GB; allow 22 GB free, plus space for the app).\n'
printf 'This tester build is not Apple-notarized. If macOS blocks opening, use System Settings → Privacy & Security → Open Anyway.\n'
/usr/bin/open "$target"
