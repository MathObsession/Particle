#!/usr/bin/env python3
"""Create a self-contained local .app and a checksummed Homebrew cask/tap.
Remote release URLs can be supplied after signing/notarization and hosting.
No model weights, model credentials, or developer-machine paths enter the app.
"""
import argparse,hashlib,json,os,plistlib,shutil,subprocess,tempfile,datetime,base64
from urllib.parse import urlparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bin',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'dist');p.add_argument('--version',default='0.3.2');p.add_argument('--build-number');p.add_argument('--release-url');p.add_argument('--homepage',default='https://huggingface.co/MathObsession');p.add_argument('--identity',default='-');p.add_argument('--update-feed');p.add_argument('--update-public-key');a=p.parse_args()
if not a.build_number:a.build_number=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
channel=json.loads((ROOT/'Scripts/particleone/update-channel.json').read_text())
feed=a.update_feed or channel['feed_url'];public_key=a.update_public_key or channel['public_key']
parsed=urlparse(feed)
if not (parsed.scheme=='https' or (parsed.scheme=='http' and parsed.hostname=='127.0.0.1')) or parsed.username or parsed.password:raise SystemExit('Update feed must use HTTPS or the local 127.0.0.1 development host.')
if len(base64.b64decode(public_key,validate=True))!=32:raise SystemExit('Invalid update public key.')
a.output.mkdir(parents=True,exist_ok=True)
def run(*args):subprocess.run([str(x) for x in args],check=True)
help_text=subprocess.run([str(a.bin/'ParticleCLI'),'--help'],capture_output=True,text=True,check=True).stdout
if '--desktop-events' not in help_text or '--reasoning-budget' not in help_text:raise SystemExit('Bundled engine lacks the desktop transport. Rebuild ParticleCLI.')
for name in ['ParticleOne','ParticleCLI']:
 if not (a.bin/name).is_file():raise SystemExit('Build '+name+' first.')
app=a.output/'Particle.app'
if app.exists():
 info=app/'Contents/Info.plist'
 if not info.is_file() or plistlib.loads(info.read_bytes()).get('CFBundleIdentifier')!='app.particleone.desktop':raise SystemExit('Refusing to replace unrelated app.')
 # Only replace this packaging output, never a running or installed application.
 processes=subprocess.run(['/bin/ps','-axww','-o','comm='],capture_output=True,text=True,check=True).stdout
 if str(app.resolve())+'/Contents/MacOS/' in processes:raise SystemExit('Close this ParticleOne build before packaging again.')
with tempfile.TemporaryDirectory(prefix='particleone-stage-',dir=a.output) as temp:
 stage=Path(temp)/'Particle.app';exe=stage/'Contents/MacOS';res=stage/'Contents/Resources';exe.mkdir(parents=True);res.mkdir(parents=True)
 for name in ['ParticleOne','ParticleCLI']:shutil.copy2(a.bin/name,exe/name)
 bundles=['Particle_ParticleOne.bundle','Particle_ParticleCore.bundle','Particle_TurboFieldfare.bundle','swift-transformers_Hub.bundle','swift-crypto_Crypto.bundle','SwiftMath_SwiftMath.bundle']
 for name in bundles:
  source=a.bin/name
  if not source.is_dir():raise SystemExit('Missing runtime resource bundle: '+name)
  for child in source.rglob('*'):
   if child.is_symlink():raise SystemExit('Resource symlink is not portable: '+str(child))
   if child.suffix in ['.gguf','.safetensors','.gturbo','.particle']:raise SystemExit('Model weights cannot be packaged.')
  shutil.copytree(source,res/name)
 # Compile the Icon Composer document so macOS uses the full tile rather
 # than shrinking a legacy square into a synthesized enclosure.
 framework=a.bin/'Sparkle.framework'
 if not framework.is_dir():raise SystemExit('Missing Sparkle.framework; rebuild ParticleOne.')
 shutil.copy2(ROOT/'Scripts/particleone/Sparkle-LICENSE',res/'Sparkle-LICENSE')
 frameworks=stage/'Contents/Frameworks';frameworks.mkdir()
 shutil.copytree(framework,frameworks/'Sparkle.framework',symlinks=True)
 icon_output=Path(temp)/'native-icons';icon_output.mkdir()
 icon_info=Path(temp)/'icon-info.plist'
 run('/usr/bin/xcrun','actool',ROOT/'Design/ParticleOne.icon','--compile',icon_output,'--platform','macosx','--minimum-deployment-target','26.0','--app-icon','ParticleOne','--output-partial-info-plist',icon_info,'--target-device','mac')
 for name in ['Assets.car','ParticleOne.icns']:
  if not (icon_output/name).is_file():raise SystemExit('Native icon compilation failed: '+name)
  shutil.copy2(icon_output/name,res/name)
 icon_values=plistlib.loads(icon_info.read_bytes())
 values=dict(CFBundleExecutable='ParticleOne',CFBundleIdentifier='app.particleone.desktop',CFBundleName='Particle',CFBundleDisplayName='Particle',CFBundlePackageType='APPL',CFBundleShortVersionString=a.version,CFBundleVersion=a.build_number or a.version,LSMinimumSystemVersion='26.0',CFBundleIconFile='ParticleOne.icns',NSHighResolutionCapable=True,NSPrincipalClass='NSApplication',NSSupportsAutomaticTermination=False)
 values.update(icon_values)
 values.update(SUFeedURL=feed,SUPublicEDKey=public_key,SURequireSignedFeed=True,SUVerifyUpdateBeforeExtraction=True,SUEnableAutomaticChecks=False,SUAutomaticallyUpdate=False,SUAllowsAutomaticUpdates=False,SUEnableSystemProfiling=False,SUShowReleaseNotes=False)
 if parsed.scheme=='http':values['NSAppTransportSecurity']={'NSAllowsLocalNetworking':True}
 (stage/'Contents/Info.plist').write_bytes(plistlib.dumps(values));run('/usr/bin/plutil','-lint',stage/'Contents/Info.plist')
 if a.identity!='-':
  # Sign nested code from the inside out for distribution, retaining Sparkle entitlements.
  nested=frameworks/'Sparkle.framework/Versions/B'
  for relative in ['XPCServices/Downloader.xpc','XPCServices/Installer.xpc','Autoupdate','Updater.app']:
   target=nested/relative
   if target.exists():run('/usr/bin/codesign','--force','--sign',a.identity,'--preserve-metadata=entitlements',target)
  run('/usr/bin/codesign','--force','--sign',a.identity,frameworks/'Sparkle.framework')
 for name in ['ParticleCLI','ParticleOne']:run('/usr/bin/codesign','--force','--sign',a.identity,exe/name)
 run('/usr/bin/codesign','--force','--sign',a.identity,stage);run('/usr/bin/codesign','--verify','--deep','--strict',stage)
 if app.exists():shutil.rmtree(app)
 shutil.move(stage,app)
archive=a.output/f'Particle-{a.version}-arm64.zip'
if archive.exists():archive.unlink()
run('/usr/bin/ditto','-c','-k','--sequesterRsrc','--keepParent',app,archive)
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
url=a.release_url or archive.resolve().as_uri()
tap=a.output/'homebrew-particleone';(tap/'Casks').mkdir(parents=True,exist_ok=True)
recipe=f'''cask "particleone" do
  version "{a.version}"
  sha256 "{sha}"

  url "{url}"
  name "Particle"
  desc "Local Churo chat and a coding agent for your Mac"
  homepage "{a.homepage}"

  depends_on arch: :arm64
  depends_on macos: :tahoe

  app "Particle.app"

  zap trash: "~/Library/Application Support/ParticleOne"
end
'''
(tap/'Casks/particleone.rb').write_text(recipe)
(tap/'README.md').write_text('# ParticleOne local Homebrew tap\n\nGenerated from the checksummed local app archive. Replace the file URL with a signed/notarized public release URL for distribution. Model weights are downloaded separately.\n')
if not (tap/'.git').exists():run('git','init','-q',tap)
run('git','-C',tap,'add','Casks/particleone.rb','README.md')
changed=subprocess.run(['git','-C',str(tap),'diff','--cached','--quiet']).returncode
if changed:run('git','-C',tap,'-c','user.name=ParticleOne Build','-c','user.email=build@particleone.local','commit','-q','-m',f'Package ParticleOne {a.version}')
report=dict(app_name="Particle",version=a.version,update_feed=feed,update_public_key=public_key,build=a.build_number or a.version,icon_format='Icon Composer / Assets.car',app=str(app),archive=str(archive),sha256=sha,cask=str(tap/'Casks/particleone.rb'),signing='ad-hoc local development' if a.identity=='-' else 'custom identity',notarized=False,publicly_hosted=bool(a.release_url),install_commands=[f'brew tap --custom-remote particleone/local {tap.resolve().as_uri()}','brew install --cask particleone/local/particleone'])
(a.output/'particleone-release.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
