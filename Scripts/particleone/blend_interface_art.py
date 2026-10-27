"""Blend the supplied ParticleDesign PNG edges once; SwiftUI displays the PNG.
The original supplied hero and its colored background are preserved. No live blur, shape approximation,
new font, or replacement logo is used in the banner.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageFilter
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'Sources/ParticleOne/Resources/Art'
OUT=ROOT/'docs/particleone/validation-0.3.0'
OUT.mkdir(parents=True,exist_ok=True)
original=Image.open(ART/'hero.png').convert('RGBA')
w,h=original.size
# Preserve the exact center of the supplied artwork at a wide UI aspect.
height=round(w/3)
source=original.crop((0,(h-height)//2,w,(h+height)//2))
y,x=np.mgrid[0:height,0:w].astype(np.float32)
u=x/(w-1);v=y/(height-1)
def smooth(q):
 q=np.clip(q,0,1)
 return q*q*q*(q*(q*6-15)+10)
# Keep the PNG background. Only tint the outside toward the app color.
# A C2 transition keeps the center wash and all lettering at original strength.
fx=smooth(np.minimum(u,1-u)/0.12)
fy=smooth(np.minimum(v,1-v)/0.34)
blend=fx*fy
pixels=np.asarray(source.convert('RGB'),dtype=np.float32)
reports={}
for name,paper in [('light',(247,248,242)),('dark',(32,36,31))]:
 rgb=pixels*blend[...,None]+np.array(paper,dtype=np.float32)*(1-blend[...,None])
 image=Image.fromarray(np.uint8(np.clip(np.rint(rgb),0,255)),'RGB')
 path=ART/f'hero-blended-{name}.png';image.save(path)
 preview=image.copy();preview.thumbnail((1500,1500));preview.save(OUT/f'banner-{name}.jpg',quality=95)
 reports[name]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=image.size,background='#%02X%02X%02X'%paper,mode='opaque PNG: original background preserved, edge color blended',outer_pixel=image.getpixel((0,0)))
# True alpha avoids luma filters, background tiles, and compositing artifacts.
mask=Image.open(ART/'icon-mask.png').convert('L')
icon=Image.new('RGBA',mask.size,(255,255,255,0));icon.putalpha(mask);icon.save(ART/'icon-alpha.png')
(OUT/'artwork.json').write_text(json.dumps(dict(original_hero_sha256=hashlib.sha256((ART/'hero.png').read_bytes()).hexdigest(),assets=reports),indent=2))
print(json.dumps(reports,indent=2))
