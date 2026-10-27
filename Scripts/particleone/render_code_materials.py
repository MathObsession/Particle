#!/usr/bin/env python3
"""ParticleDesign's organic wash/grain as native coding-workspace PNG materials.
Deterministic, code-rendered; exact paper edges keep the art inside the app.
"""
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'Sources/ParticleOne/Resources/Art';OUT=ROOT/'docs/particleone/validation-0.3.1'
OUT.mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[0:768,0:1536].astype(np.float32);u=x/1535;v=y/767
rng=np.random.default_rng(310)
a=u+.07*np.sin(5*v+2*u);b=v+.09*np.sin(5*u-.8)
def cloud(cx,cy,sx,sy):return np.exp(-.5*(((a-cx)/sx)**2+((b-cy)/sy)**2))[...,None]
def smooth(q):q=np.clip(q,0,1);return q*q*q*(q*(q*6-15)+10)
edge=(smooth(np.minimum(u,1-u)/.16)*smooth(np.minimum(v,1-v)/.25))[...,None]
noise=rng.normal(0,1.35,(*u.shape,1))
for name,base,green,mint in [('light',(247,248,242),(221,234,199),(224,236,229)),('dark',(32,36,31),(51,62,44),(38,49,45))]:
 field=np.empty((*u.shape,3),dtype=np.float32);field[:]=base
 for color,cx,cy,sx,sy,strength in [(green,.65,.3,.38,.42,.78),(mint,.14,.85,.4,.4,.55)]:
  q=cloud(cx,cy,sx,sy)*strength;field=field*(1-q)+np.array(color)*q
 # Faint off-canvas arcs stay subordinate to interface text.
 radius=np.sqrt(((u+0.15)/1.0)**2+((v-.4)/1.3)**2)
 arc=np.exp(-((radius-.9)/.0025)**2)[...,None]*.7
 field=np.array(base)+(field-np.array(base)+noise+arc)*edge
 image=Image.fromarray(np.uint8(np.clip(np.rint(field),0,255)))
 image.save(ART/f'code-scene-{name}.png');image.resize((1000,500)).save(OUT/f'code-material-{name}.jpg',quality=95)
 # Denser material for composer/action cards; quiet, grainy and readable.
 panel=np.array(base)+(field-np.array(base))*.65+(3 if name=='dark' else -2)
 Image.fromarray(np.uint8(np.clip(np.rint(panel),0,255))).save(ART/f'code-panel-{name}.png')
# Seamless fine-grain overlay for broader app surfaces.
noise=np.uint8(rng.integers(0,256,size=(256,256)))
rgba=np.zeros((256,256,4),dtype=np.uint8);rgba[:,:,:3]=noise[...,None];rgba[:,:,3]=10
Image.fromarray(rgba).save(ART/'interface-grain.png')
print('Rendered code-scene, code-panel and interface-grain PNGs.')
