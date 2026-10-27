#!/usr/bin/env python3
"""Publish a portable .particle pack. Authentication is transient and never bundled."""
import argparse,datetime,getpass,hashlib,json,os,sys
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd,get_token
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pack',type=Path,default=Path('/Volumes/Particle/Models/sarvam-30b/sarvam-30b-q4_k_m.particle'))
p.add_argument('--repo',default='MathObsession/sarvam-30b-particle')
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'docs/particleone/model-release')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def status(stage,**fields):
 record=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=a.repo,**fields)
 (a.output/'status.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 token=os.environ.get('HF_TOKEN') or get_token() or getpass.getpass('Hugging Face write token (hidden, not saved): ')
 api=HfApi(token=token);identity=api.whoami()
 if identity.get('name','').lower()!=a.repo.split('/')[0].lower():raise RuntimeError('Account does not match repository owner')
 status('authenticated',account=identity['name'])
 pack=a.pack.resolve();manifest=json.loads((pack/'manifest.json').read_text());receipt=json.loads((pack/'verified-install.json').read_text());manifest_sha=digest(pack/'manifest.json')
 if not receipt.get('complete') or receipt.get('manifestSHA256')!=manifest_sha:raise RuntimeError('Local verification receipt mismatch')
 if manifest.get('profile')!='native' or manifest.get('approximate') or manifest.get('architecture')!='sarvam_moe':raise RuntimeError('Expected native Sarvam pack')
 relative=['manifest.json','weights.json','vocabulary.json']+sorted(manifest['files'])
 relative+=sorted(str(f.relative_to(pack)) for f in (pack/'tokenizer').rglob('*') if f.is_file() and f.suffix in ['.json','.jinja'])
 operations=[];files=[];prefix=pack.name
 for i,name in enumerate(sorted(set(relative))):
  path=pack/name
  if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(pack):raise RuntimeError('Non-portable source')
  status('hashing',file=name,file_index=i+1,file_count=len(set(relative)),bytes=path.stat().st_size)
  op=CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=path);sha=op.upload_info.sha256.hex();size=op.upload_info.size
  expected=manifest['files'].get(name)
  if expected and (expected['sha256']!=sha or expected['bytes']!=size):raise RuntimeError('Pack checksum mismatch')
  operations.append(op);files.append(dict(path=name,bytes=size,sha256=sha))
 distribution=dict(version=1,format='particle',folder=prefix,architecture='sarvam_moe',profile='native',approximate=False,manifestSHA256=manifest_sha,files=files,totalBytes=sum(f['bytes'] for f in files))
 data=(json.dumps(distribution,indent=2,sort_keys=True)+'\n').encode();(a.output/'particle-distribution.json').write_bytes(data)
 card='''---\nlicense: apache-2.0\nbase_model: sarvamai/sarvam-30b\ntags:\n- particle\n- sarvam\n- quantized\n- apple-silicon\n---\n# Sarvam 30B · Particle native pack\n\nA lossless runtime-layout conversion of Q4_K_M Sarvam for ParticleOne on Apple Silicon. This `.particle` directory retains native Q4_K/Q5_K/Q6_K bytes; it is not a renamed GGUF or a new quantization.\n\nOriginal: [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b), Apache 2.0. Quantization: [mtrajan/sarvam-30b-GGUF](https://huggingface.co/mtrajan/sarvam-30b-GGUF), `sarvam-30b-q4_k_m.gguf`, SHA-256 `411e26bd027f280961e821dc5d25b605be9195ae8f17d8f89ad1258302aa4948`. Tokenizer/chat template: official Sarvam repository.\n\n`particle-distribution.json` lists each downloadable file, size and SHA-256. Download into the named folder and verify every file before creating a local `verified-install.json` receipt. Local progress, locks, logs, personal paths and credentials are excluded.\n\nThe default graph uses six experts; lower expert counts are explicit runtime approximations. No universal speed or quality-loss guarantee is claimed. ParticleOne requires Apple Silicon and macOS 26+.\n'''
 root=Path(__file__).resolve().parents[2];license_path=root/'Sources/ParticleOne/Resources/Model-APACHE-2.0.txt'
 if not license_path.is_file():raise RuntimeError('Model license is missing')
 operations.extend([CommitOperationAdd(path_in_repo='particle-distribution.json',path_or_fileobj=data),CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()),CommitOperationAdd(path_in_repo='LICENSE',path_or_fileobj=license_path)])
 api.create_repo(a.repo,repo_type='model',private=False,exist_ok=True)
 status('uploading',file_count=len(files),total_bytes=distribution['totalBytes'])
 commit=api.create_commit(a.repo,repo_type='model',operations=operations,commit_message='Publish verified native Sarvam 30B Particle pack',num_threads=2)
 remote={f.rfilename:f for f in api.model_info(a.repo,revision=commit.oid,files_metadata=True).siblings}
 for f in files:
  rf=remote.get(prefix+'/'+f['path'])
  if rf is None or rf.size!=f['bytes'] or (rf.lfs and rf.lfs.sha256!=f['sha256']):raise RuntimeError('Remote file verification failed')
 catalog=dict(repo=a.repo,revision=commit.oid,distributionSHA256=hashlib.sha256(data).hexdigest(),modelManifestSHA256=manifest_sha,folder=prefix,totalBytes=distribution['totalBytes'])
 for target in [root/'Sources/ParticleOne/Resources/model-catalog.json',a.output/'catalog.json']:target.write_text(json.dumps(catalog,indent=2)+'\n')
 status('published',revision=commit.oid,url='https://huggingface.co/'+a.repo,total_bytes=distribution['totalBytes'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  status('failed',error_type=type(e).__name__,message='Publishing failed. Check account write permissions, connectivity and pack checksums. No token was saved.');sys.exit(1)
