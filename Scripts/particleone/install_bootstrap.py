#!/usr/bin/env python3
"""Install the first updater-enabled copy at one stable per-user location."""
import argparse,os,plistlib,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'dist/update-staging/Particle.app');a=p.parse_args()
    target=Path.home()/'Applications/Particle.app'
    def validate(path):
        info=plistlib.loads((path/'Contents/Info.plist').read_bytes())
        if info.get('CFBundleIdentifier')!='app.particleone.desktop' or not info.get('SUPublicEDKey'):raise SystemExit('Not an updater-enabled ParticleOne app.')
        subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(path)],check=True)
    validate(a.source)
    processes=subprocess.check_output(['/bin/ps','-axww','-o','comm='],text=True)
    if str(target)+'/Contents/MacOS/' in processes:raise SystemExit('ParticleOne is already running at its stable location. Use its in-app update button.')
    target.parent.mkdir(exist_ok=True)
    if target.exists():raise SystemExit('A stable copy is already installed. Use Check for updates in that app; it will be replaced in place.')
    with tempfile.TemporaryDirectory(prefix='.particle-install-',dir=target.parent) as folder:
        staged=Path(folder)/'Particle.app';shutil.copytree(a.source,staged,symlinks=True);validate(staged);os.rename(staged,target)
    print('Installed updater-enabled ParticleOne at '+str(target))
    print('Quit any older open copy, then open this app once. Future updates replace it in place.')
if __name__=='__main__':main()
