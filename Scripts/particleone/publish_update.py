#!/usr/bin/env python3
"""Sign and publish an app-only release to this Mac's persistent local update feed."""
import argparse,hashlib,json,os,plistlib,re,shutil,subprocess,sys,tempfile,time,urllib.request,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from cleanup_releases import prune_archives,archive_folders
ROOT=Path(__file__).resolve().parents[2]
NS='http://www.andymatuschak.org/xml-namespaces/sparkle'
ET.register_namespace('sparkle',NS)

def validate_appcast(data,release,archive_url,length):
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)*',release['build']):raise ValueError('Build must be a numeric version.')
    root=ET.fromstring(data);item=root.find('.//item')
    if item is None or item.findtext('{'+NS+'}version')!=release['build'] or item.findtext('{'+NS+'}shortVersionString')!=release['version']:raise ValueError('Generated feed has the wrong version.')
    entry=item.find('enclosure')
    if entry is None or entry.attrib.get('url')!=archive_url or int(entry.attrib.get('length','0'))!=length:raise ValueError('Generated feed has the wrong archive.')
    return entry.attrib['{'+NS+'}edSignature']

def refresh_local_metadata(release,target):
    tap=ROOT/'dist/homebrew-particleone';recipe=Path(release['cask']).read_text()
    recipe=recipe.replace(Path(release['archive']).resolve().as_uri(),target.resolve().as_uri())
    (tap/'Casks').mkdir(parents=True,exist_ok=True);(tap/'Casks/particleone.rb').write_text(recipe)
    if (tap/'.git').exists():
        subprocess.run(['git','-C',str(tap),'add','Casks/particleone.rb'],check=True)
        if subprocess.run(['git','-C',str(tap),'diff','--cached','--quiet']).returncode:
            subprocess.run(['git','-C',str(tap),'-c','user.name=ParticleOne Build','-c','user.email=build@particleone.local','commit','-q','-m','Publish ParticleOne '+release['version']+' ('+release['build']+')'],check=True)
    receipt=dict(release);receipt['archive']=str(target);receipt['cask']=str(tap/'Casks/particleone.rb')
    receipt['installed_app']=str(Path.home()/'Applications/Particle.app')
    receipt['install_commands']=['brew tap --custom-remote particleone/local '+tap.resolve().as_uri(),'brew install --cask particleone/local/particleone']
    (ROOT/'dist/particleone-release.json').write_text(json.dumps(receipt,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,default=ROOT/'dist/update-staging/particleone-release.json');p.add_argument('--notes',default='Improvements to ParticleOne.');p.add_argument('--no-serve',action='store_true');a=p.parse_args()
    release=json.loads(a.release.read_text());feed=urlparse(release['update_feed'])
    if feed.scheme!='http' or feed.hostname!='127.0.0.1' or feed.port!=49271 or feed.path!='/appcast.xml':raise SystemExit('This command publishes only the local channel. Host a signed appcast separately for public HTTPS channels.')
    archive=Path(release['archive']);digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if digest!=release['sha256']:raise SystemExit('Archive checksum mismatch.')
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',release['app']],check=True)
    channel=json.loads((ROOT/'Scripts/particleone/update-channel.json').read_text())
    tools=Path('/Volumes/Particle/Build/Particle/SwiftPM/artifacts/sparkle/Sparkle/bin')
    home=Path.home()/'Library/Application Support/ParticleOneUpdates';destination=home/'Releases';destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    existing=destination/'appcast.xml'
    if existing.exists():
        old=ET.fromstring(existing.read_bytes()).find('.//{'+NS+'}version')
        if old is not None and old.text and tuple(map(int,release['build'].split('.')))<=tuple(map(int,old.text.split('.'))):raise SystemExit('Refusing to publish a non-increasing build number.')
    name=release.get('app_name','ParticleOne')+'-'+release['version']+'-'+release['build']+'-arm64.zip'
    target=destination/name
    if target.exists() and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()!=digest:raise SystemExit('Release archive already exists with different bytes.')
    # One Keychain access signs the archive and feed. Verification needs only
    # the public key, never another access request for the private signing key.
    with tempfile.TemporaryDirectory(prefix='.publish-',dir=home) as folder:
        staged=Path(folder);staged_archive=staged/name;shutil.copy2(archive,staged_archive)
        staged_archive.with_suffix('.txt').write_text(a.notes)
        subprocess.run([str(tools/'generate_appcast'),'--account',channel['keychain_account'],'--maximum-deltas','0','--download-url-prefix','http://127.0.0.1:49271/','--embed-release-notes',str(staged)],check=True)
        data=(staged/'appcast.xml').read_bytes()
        signature_text=validate_appcast(data,release,'http://127.0.0.1:49271/'+name,staged_archive.stat().st_size)
        import base64
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        signature=base64.b64decode(signature_text,validate=True)
        Ed25519PublicKey.from_public_bytes(base64.b64decode(release['update_public_key'],validate=True)).verify(signature,staged_archive.read_bytes())
        if not target.exists():os.replace(staged_archive,target)
        temp=destination/'appcast.pending.xml';temp.write_bytes(data);os.replace(temp,existing)
    if not a.no_serve:
        host=home/'serve_updates.py';source_host=ROOT/'Scripts/particleone/serve_updates.py';restart_host=host.exists() and host.read_bytes()!=source_host.read_bytes();shutil.copy2(source_host,host)
        agents=Path.home()/'Library/LaunchAgents';agents.mkdir(exist_ok=True)
        plist=agents/'app.particleone.update-host.plist'
        job={'Label':'app.particleone.update-host','ProgramArguments':[sys.executable,str(host),'--directory',str(destination)],'RunAtLoad':True,'KeepAlive':True,'StandardOutPath':str(home/'host.log'),'StandardErrorPath':str(home/'host-error.log'),'ThrottleInterval':15}
        plist.write_bytes(plistlib.dumps(job));domain='gui/'+str(os.getuid());label=domain+'/app.particleone.update-host'
        loaded=subprocess.run(['/bin/launchctl','print',label],capture_output=True).returncode==0
        if not loaded:subprocess.run(['/bin/launchctl','bootstrap',domain,str(plist)],check=True)
        subprocess.run(['/bin/launchctl','kickstart']+(['-k'] if loaded and restart_host else [])+[label],check=True)
        for attempt in range(25):
            try:
                with urllib.request.urlopen(release['update_feed'],timeout=2) as response:served=response.read()
                if served==existing.read_bytes():break
            except OSError:pass
            time.sleep(0.2)
        else:raise SystemExit('Signed release saved, but local update host is unavailable or serving another feed. Check the host log; no unrelated process was stopped.')
    refresh_local_metadata(release,target)
    prune_archives(archive_folders(ROOT,Path.home()),release['version'],release['build'])
    print('Published Particle '+release['version']+' ('+release['build']+') to '+release['update_feed'])
    print('Existing updater-enabled installations will find it at their next check.')
if __name__=='__main__':main()
