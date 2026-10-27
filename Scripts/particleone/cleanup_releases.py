#!/usr/bin/env python3
"""Keep only the current Particle release; never touch models or workspace data."""
from pathlib import Path
import argparse,json,plistlib,re,shutil,subprocess

BUNDLE_ID='app.particleone.desktop'
ARCHIVE=re.compile(r'^(?:[0-9a-f]{64}--)?Particle(?:One)?-([0-9]+(?:\.[0-9]+)*)(?:-([0-9]+))?-arm64\.zip$')

def number(value):
    if not isinstance(value,str) or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)*',value):raise ValueError('Invalid release number')
    parts=list(map(int,value.split('.')))
    while len(parts)>1 and parts[-1]==0:parts.pop()
    return tuple(parts)

def older(version,build,current_version,current_build):
    a,b=number(version),number(current_version)
    return a<b or (a==b and build is not None and number(build)<number(current_build))

def prune_archives(folders,version,build):
    number(version);number(build);removed=[]
    for folder in folders:
        if not folder.is_dir() or folder.is_symlink():continue
        for path in folder.iterdir():
            match=ARCHIVE.fullmatch(path.name)
            if path.is_symlink() or not path.is_file() or not match:continue
            if older(match[1],match[2],version,build):path.unlink();removed.append(str(path))
    return removed

def archive_folders(root,home):
    return [root/'dist',root/'dist/update-staging',home/'Library/Application Support/ParticleOneUpdates/Releases',home/'Library/Caches/Homebrew/downloads']

def cleanup(root,version,build,home=None):
    home=home or Path.home();app=home/'Applications/Particle.app'
    current=plistlib.loads((app/'Contents/Info.plist').read_bytes())
    if current.get('CFBundleIdentifier')!=BUNDLE_ID or current.get('CFBundleShortVersionString')!=version or current.get('CFBundleVersion')!=build:raise RuntimeError('Install the current release before pruning application copies.')
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],check=True)
    processes=subprocess.check_output(['/bin/ps','-axww','-o','comm='],text=True).splitlines()
    removed=prune_archives(archive_folders(root,home),version,build);active=[]
    folders=[root/'dist',root/'dist/update-staging',home/'Applications',Path('/Applications'),home/'.Trash']
    cache=home/'Library/Caches/Sparkle_generate_appcast'
    candidates=[]
    for folder in folders:
        if folder.is_symlink():continue
        try:candidates.extend(folder.glob('*Particle*.app'))
        except OSError:continue
    if cache.is_dir() and not cache.is_symlink():candidates.extend(cache.glob('*/*Particle*.app'))
    for path in candidates:
        if path.is_symlink() or not path.is_dir() or path==app:continue
        try:info=plistlib.loads((path/'Contents/Info.plist').read_bytes())
        except (OSError,ValueError):continue
        if info.get('CFBundleIdentifier')!=BUNDLE_ID:continue
        try:is_old=older(info.get('CFBundleShortVersionString'),info.get('CFBundleVersion'),version,build)
        except ValueError:continue
        if not is_old:continue
        if any(str(path)+'/Contents/' in line for line in processes):active.append(str(path));continue
        shutil.rmtree(path);removed.append(str(path))
    return {'current_version':version,'current_build':build,'removed':removed,'active_older_copies':active}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args()
    receipt=json.loads((args.root/'dist/particleone-release.json').read_text())
    report=cleanup(args.root,receipt['version'],receipt['build']);print(json.dumps(report,indent=2))
    output=args.root/('docs/particleone/validation-'+receipt['version']);output.mkdir(parents=True,exist_ok=True)
    (output/'older-release-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['active_older_copies']:raise SystemExit('Close the listed older Particle copies normally, then rerun cleanup.')
if __name__=='__main__':main()
