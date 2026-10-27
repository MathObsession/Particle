"""Prepare the full-canvas ParticleDesign Icon Composer source.
The macOS compiler supplies the outer rounded mask; artwork has no inset tile.
"""
from pathlib import Path
import json
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
output=ROOT/'Design/ParticleOne.icon'
(output/'Assets').mkdir(parents=True,exist_ok=True)
Image.open(ROOT/'Sources/ParticleOne/Resources/Art/icon.png').convert('RGB').save(output/'Assets/Artwork.png')
document={"fill":{"solid":"srgb:0.69020,0.83529,0.56078,1.00000"},"groups":[{"name":"ParticleDesign artwork","layers":[{"name":"Full canvas artwork","image-name":"Artwork.png","glass":False,"position":{"scale":1,"translation-in-points":[0,0]}}],"shadow":{"kind":"none","opacity":0},"specular":False,"translucency":{"enabled":False,"value":0}}],"supported-platforms":{"squares":["macOS"]}}
(output/'icon.json').write_text(json.dumps(document,indent=2)+'\n')
print(output)
