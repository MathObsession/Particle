#!/usr/bin/env python3
"""Promote this Mac's verified SSD mirror into the permanent local model.
Reuse the existing payload inodes; copy only missing sidecars/embeddings.
No source model or developer tools on the SD card are removed.
"""
from pathlib import Path
import copy,datetime,fcntl,hashlib,json,os,shutil,sqlite3,stat,subprocess
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle')
SUPPORT=Path.home()/'Library/Application Support/ParticleOne'
TARGET=SUPPORT/'Models/sarvam-30b-q4_k_m.particle'
CACHE=Path.home()/'Library/Caches/Particle/Experts'
MIRROR=CACHE/'mirror-v1'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def no_running_app():
    lines=subprocess.check_output(['/bin/ps','-axww','-o','comm='],text=True).splitlines()
    if any('/ParticleOne.app/Contents/MacOS/' in s or Path(s.strip()).name in ['ParticleCLI','TurboFieldfareCLI','TurboFieldfareMac','TurboFieldfareServer'] for s in lines):raise RuntimeError('Close ParticleOne and any Particle model session before migrating. No process was stopped.')

def main():
    no_running_app()
    if TARGET.exists():raise RuntimeError('The permanent model already exists; it has not been overwritten.')
    manifest_data=(SOURCE/'manifest.json').read_bytes();manifest=json.loads(manifest_data)
    receipt=json.loads((SOURCE/'verified-install.json').read_bytes())
    assert receipt['complete'] and receipt['manifestSHA256']==hashlib.sha256(manifest_data).hexdigest()
    assert receipt['manifestSHA256']=='1aa97aae017f8ef868e0ba0d263a5e4f8a08b0fb8547f4741260315d5880f0e6'
    lock=os.open(CACHE/'.tier.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        mirror_receipt=json.loads((MIRROR/'mirror-receipt.json').read_bytes())
        expected={name:record for name,record in manifest['files'].items() if name=='model_weights.bin' or (name.startswith('packed_experts/layer_') and name.endswith('.bin'))}
        assert mirror_receipt['identity']=='particle-native-v1|'+receipt['manifestSHA256'] and mirror_receipt['files']==expected
        TARGET.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        partial=TARGET.with_name(TARGET.name+'.local-installing');partial.mkdir(mode=0o700,exist_ok=True)
        missing=sum(v['bytes'] for k,v in manifest['files'].items() if k not in expected)
        assert shutil.disk_usage(TARGET.parent).free>missing+2*1024**3
        copied=0;reused=0
        for name,record in sorted(manifest['files'].items()):
            relative=Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            destination=partial/relative;destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            source=MIRROR/relative.name if name in expected else SOURCE/relative
            assert stat.S_ISREG(source.lstat().st_mode) and source.stat().st_size==record['bytes']
            if destination.exists():
                assert not destination.is_symlink() and destination.stat().st_size==record['bytes'] and digest(destination)==record['sha256']
            elif name in expected:
                assert source.stat().st_dev==partial.stat().st_dev
                os.link(source,destination);reused+=record['bytes']
            else:
                shutil.copyfile(source,destination);os.chmod(destination,0o600);copied+=record['bytes']
            assert destination.stat().st_size==record['bytes'] and digest(destination)==record['sha256'],name
            print('Verified',name,flush=True)
        (partial/'manifest.json').write_bytes(manifest_data)
        (partial/'verified-install.json').write_bytes((SOURCE/'verified-install.json').read_bytes())
        assert not any(p.is_symlink() for p in partial.rglob('*'))
        for p in partial.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        os.rename(partial,TARGET)
        no_running_app()
        databases={SUPPORT/'particleone.sqlite'}
        # Include an explicitly selected workspace folder, if one is configured.
        selected=subprocess.run(['/usr/bin/defaults','read','app.particleone.desktop','ParticleOne.lastWorkspaceDirectory'],capture_output=True,text=True)
        if selected.returncode==0:
            chosen=Path(selected.stdout.strip())/'particleone.sqlite'
            if chosen.exists():databases.add(chosen)
        backup_root=SUPPORT/'MigrationBackups';backup_root.mkdir(mode=0o700,exist_ok=True)
        updated=0;backups=[]
        for database in sorted(databases):
            if not database.exists():continue
            db=sqlite3.connect(database)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            backup=backup_root/(hashlib.sha256(str(database).encode()).hexdigest()[:8]+'-'+stamp+'.sqlite')
            with sqlite3.connect(backup) as saved:db.backup(saved)
            os.chmod(backup,0o600);backups.append(str(backup))
            try:
                db.execute('BEGIN IMMEDIATE')
                for id,raw in db.execute('SELECT id,payload FROM workspaces').fetchall():
                    old=json.loads(raw);new=copy.deepcopy(old);settings=new.get('settings',{})
                    if settings.get('modelPath') not in [str(SOURCE),str(SOURCE.with_suffix('.gguf'))]:continue
                    settings['modelPath']=str(TARGET);settings['useMirror']=False
                    payload=json.dumps(new,ensure_ascii=False,separators=(',',':')).encode()
                    db.execute('UPDATE workspaces SET payload=? WHERE id=?',(payload,id));updated+=1
                    check=copy.deepcopy(new);check['settings']['modelPath']=old['settings']['modelPath'];check['settings']['useMirror']=old['settings'].get('useMirror',True)
                    old_check=copy.deepcopy(old);old_check['settings'].setdefault('useMirror',True)
                    assert check==old_check
                db.commit();assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            except Exception:db.rollback();raise
            finally:db.close()
        # Publish first, then drop old cache names. These are the same inodes,
        # so the transition never allocates a second common/expert payload.
        assert set(p.name for p in MIRROR.iterdir())==set(Path(n).name for n in expected)|{'mirror-receipt.json'}
        for name in expected:
            old=MIRROR/Path(name).name;new=TARGET/name;assert os.path.samefile(old,new);old.unlink()
        (MIRROR/'mirror-receipt.json').unlink();MIRROR.rmdir()
        report={'model':str(TARGET),'files_verified':len(manifest['files']),'total_model_bytes':sum(v['bytes'] for v in manifest['files'].values()),'reused_ssd_payload_bytes':reused,'copied_from_sd_bytes':copied,'updated_workspaces':updated,'workspace_backups':backups,'cache_promoted_without_duplicate_payload':True,'source_sd_model_preserved':True}
        out=ROOT/'docs/particleone/standalone';out.mkdir(exist_ok=True);(out/'migration.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)
if __name__=='__main__':main()
