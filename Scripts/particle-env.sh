# Source this file to use Particle's SD-card toolchain in the current shell.
export PARTICLE_SD_ROOT="${PARTICLE_SD_ROOT:-/Volumes/Particle}"
export PARTICLE_TOOLCHAIN="${PARTICLE_TOOLCHAIN:-xcode}"
case "$PARTICLE_TOOLCHAIN" in
  xcode) export DEVELOPER_DIR="$PARTICLE_SD_ROOT/Xcode.app/Contents/Developer" ;;
  clt) export DEVELOPER_DIR="/Library/Developer/CommandLineTools" ;;
  *) printf 'Unknown PARTICLE_TOOLCHAIN: %s\n' "$PARTICLE_TOOLCHAIN" >&2; return 1 2>/dev/null || exit 1 ;;
esac
if [ ! -d "$DEVELOPER_DIR" ]; then
    printf '%s\n' "Particle SD card or Xcode is unavailable: $PARTICLE_SD_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi
export PARTICLE_MODEL_PATH="${PARTICLE_MODEL_PATH:-$PARTICLE_SD_ROOT/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf}"
export PARTICLE_BUILD_ROOT="$PARTICLE_SD_ROOT/Build/Particle"
export PARTICLE_CACHE_ROOT="$PARTICLE_SD_ROOT/Caches/Particle"
export CLANG_MODULE_CACHE_PATH="$PARTICLE_CACHE_ROOT/Clang"
export SWIFTPM_MODULECACHE_OVERRIDE="$PARTICLE_CACHE_ROOT/SwiftModules"
if [ "$PARTICLE_TOOLCHAIN" = "xcode" ]; then
    export PATH="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin:$DEVELOPER_DIR/usr/bin:/usr/bin:$PATH"
else
    export PATH="$DEVELOPER_DIR/usr/bin:/usr/bin:$PATH"
fi
