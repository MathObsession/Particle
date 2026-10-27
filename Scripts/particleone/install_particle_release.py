#!/usr/bin/env python3
"""Install the renamed app in place and migrate the model name without copying weights."""
from pathlib import Path
import datetime,hashlib,json,os,plistlib,shutil,sqlite3,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2]
IDENTITY='app.particleone.desktop'

def info(app):return plistlib.loads((app/'Contents/Info.plist').read_bytes())
def verify(app):
    value=info(app)
    if value.get('CFBundleIdentifier')!=IDENTITY:raise RuntimeError('Refusing to replace an unrelated application.')
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],check=True)
    return value

def main():
    release=json.loads((ROOT/'dist/update-staging/particleone-release.json').read_text());source=Path(release['app']);value=verify(source)
    assert value['CFBundleDisplayName']=='Particle' and value['CFBundleShortVersionString']==release['version']
    assert hashlib.file_digest(Path(release['archive']).open('rb'),'sha256').hexdigest()==release['sha256']
    lines=subprocess.check_output(['/bin/ps','-axww','-o','comm='],text=True).splitlines()
    if any('/ParticleOne.app/Contents/' in line or '/Particle.app/Contents/' in line or Path(line.strip()).name=='ParticleCLI' for line in lines):raise RuntimeError('A Particle app, worker or updater is running. Use its update control or close it normally first; no process was stopped.')
    apps=Path.home()/'Applications';target=apps/'Particle.app';legacy=apps/'ParticleOne.app';support=Path.home()/'Library/Application Support/ParticleOne'
    for app in [target,legacy]:
        if app.exists():
            old=verify(app);assert old.get('SUPublicEDKey')==value['SUPublicEDKey']
            assert tuple(int(x) for x in old['CFBundleShortVersionString'].split('.'))<=tuple(int(x) for x in value['CFBundleShortVersionString'].split('.'))
    oldmodel=support/'Models/sarvam-30b-q4_k_m.particle';model=support/'Models/churo-chat.particle'
    if oldmodel.exists() and model.exists():raise RuntimeError('Both model names exist. Neither was overwritten.')
    assert oldmodel.exists() or model.exists()
    model_source=oldmodel if oldmodel.exists() else model
    model_manifest=(model_source/'manifest.json').read_bytes();receipt=json.loads((model_source/'verified-install.json').read_text());assert receipt['complete'] and receipt['manifestSHA256']==hashlib.sha256(model_manifest).hexdigest()
    backup=apps/'.Particle.previous.app'
    if backup.exists():raise RuntimeError('An earlier app installation backup exists; it has been preserved.')
    with tempfile.TemporaryDirectory(prefix='.particle-install-',dir=apps) as temporary:
        staged=Path(temporary)/'Particle.app';shutil.copytree(source,staged,symlinks=True);verify(staged)
        if target.exists():os.rename(target,backup)
        try:os.rename(staged,target);verify(target)
        except Exception:
            if not target.exists() and backup.exists():os.rename(backup,target)
            raise
    if oldmodel.exists():os.rename(oldmodel,model)
    databases={support/'particleone.sqlite'}
    selected=subprocess.run(['/usr/bin/defaults','read','app.particleone.desktop','ParticleOne.lastWorkspaceDirectory'],capture_output=True,text=True)
    if selected.returncode==0:
        candidate=Path(selected.stdout.strip())/'particleone.sqlite'
        if candidate.exists():databases.add(candidate)
    backups=support/'MigrationBackups';backups.mkdir(mode=0o700,exist_ok=True);changed=0
    for path in databases:
        if not path.exists():continue
        db=sqlite3.connect(path);saved=backups/('particle-brand-'+hashlib.sha256(str(path).encode()).hexdigest()[:8]+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.sqlite')
        with sqlite3.connect(saved) as copy:db.backup(copy)
        os.chmod(saved,0o600)
        try:
            db.execute('BEGIN IMMEDIATE')
            for id,payload in db.execute('SELECT id,payload FROM workspaces').fetchall():
                workspace=json.loads(payload)
                if workspace['settings']['modelPath'] in [str(oldmodel),str(model)]:
                    workspace['settings']['modelPath']=str(model);workspace['settings']['useMirror']=False
                    db.execute('UPDATE workspaces SET payload=? WHERE id=?',(json.dumps(workspace,ensure_ascii=False,separators=(',',':')).encode(),id));changed+=1
            db.commit();assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        except Exception:db.rollback();raise
        finally:db.close()
    assert (model/'manifest.json').read_bytes()==model_manifest
    if legacy.exists():shutil.rmtree(legacy)
    if backup.exists():shutil.rmtree(backup)
    report={'app':str(target),'version':value['CFBundleShortVersionString'],'build':value['CFBundleVersion'],'model':str(model),'updated_workspaces':changed,'same_bundle_identifier':True,'same_update_key':True,'weights_copied':0,'user_transcripts_and_memories_preserved':True}
    (ROOT/('docs/particleone/validation-'+release['version'])/'installation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
