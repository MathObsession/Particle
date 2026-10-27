#!/usr/bin/env python3
"""Fetch pinned tokenizer metadata only; never download model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"
HASHES = {
    "config.json": "db5a70c723fc5bd1ca123ab94d1c0a99d914a20c25eefcf041d137906e2341e9",
    "tokenizer.json": "a574ceaaff7c7a8f091179c53fd17ae33567089c099d4ff37d4cb3bc1a87e80e",
    "tokenizer_config.json": "3cdb41a3bcbc8a7e14dbcf8987ccad0cbfe8f82091f2a2c82020f1d0e219cdd5",
    "chat_template.jinja": "07a8ca6f96384a6878c9aa6d4e0616ae57f5937fa8d0ff014855704fa26a0ef4",
    "generation_config.json": "39add3da02bdcab42533aaeb36493e2041a5d18b5e3ada6baddc3daf8105e3cc",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("PARTICLE_MODEL_PATH", "/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.gguf")).parent / "tokenizer")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        target = args.output / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified {name}")
            continue
        url = f"https://huggingface.co/sarvamai/sarvam-30b/resolve/{REVISION}/{name}"
        temporary = None
        try:
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=args.output, prefix=f".{name}.", delete=False) as out:
                temporary = Path(out.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    total = 0
                    while chunk := response.read(512 * 1024):
                        total += len(chunk)
                        if total > 50 * 1024 * 1024:
                            raise RuntimeError(f"Unexpectedly large metadata file: {name}")
                        digest.update(chunk)
                        out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Tokenizer checksum mismatch: {name}")
            temporary.replace(target)
            print(f"Installed {name}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    (args.output / "source.json").write_text(json.dumps({"repo": "sarvamai/sarvam-30b", "revision": REVISION}, indent=2) + "\n")


if __name__ == "__main__":
    main()
