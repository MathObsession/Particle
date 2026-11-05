import {readFile,writeFile,mkdir,cp,rm} from 'node:fs/promises';
const root=new URL('./',import.meta.url),out=new URL('./dist/',root);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const models=JSON.parse(await readFile(new URL('models.json',root),'utf8'));
if(models.length!==14||models.filter(m=>m.group==='default').length!==1||models.filter(m=>m.group==='fast').length!==3)throw Error('Unexpected model collection');
const cards=models.map((m,i)=>`<a class="model-card ${m.group==='default'?'model-default':''}" data-model data-group="${m.group}" data-search="${escape(m.name.toLowerCase())}" href="${escape(m.url)}" target="_blank" rel="noopener noreferrer" aria-label="${escape(m.name)} — view model source (opens a new tab)"><div class="model-card-top"><span class="model-glyph" aria-hidden="true">${m.group==='fast'?'ϟ':'✳'}</span><span class="model-tag">${m.group==='default'?'DEFAULT':m.group==='fast'?'FAST':'MoE'}</span><svg class="icon model-arrow" aria-hidden="true"><use href="#arrow-up-right"/></svg></div><h3>${escape(m.name)}</h3><p>${m.group==='default'?'The original. Your everyday companion for thoughtful chat and hands-on coding.':m.group==='fast'?'A lighter active workload, with the same familiar workspace.':'Another perspective, in the same private, local workspace.'}</p><div class="model-meta"><span>${escape(m.quantization)}</span><span>${escape(m.capability)}</span></div></a>`).join('\n');
let html=(await readFile(new URL('index.html',root),'utf8')).replace('<!-- MODEL_CARDS -->',cards);
const host=process.env.VERCEL_PROJECT_PRODUCTION_URL||process.env.VERCEL_URL;
if(host&&/^[a-zA-Z0-9.-]+$/.test(host)){const origin='https://'+host;html=html.replace('content="/assets/particle-wordmark.webp"',`content="${origin}/assets/particle-wordmark.webp"`).replace('</head>',`<link rel="canonical" href="${origin}/"><meta property="og:url" content="${origin}/"></head>`);}
await rm(out,{recursive:true,force:true});await mkdir(out,{recursive:true});
for(const file of ['app.js','theme.js','style.css','models.json','favicon.ico','robots.txt','404.html'])await cp(new URL(file,root),new URL(file,out));
for(const folder of ['assets','fonts'])await cp(new URL(folder,root),new URL(folder+'/',out),{recursive:true,filter:source=>!source.endsWith('.png')||source.endsWith('logo.png')});
await writeFile(new URL('index.html',out),html);
console.log('Built Particle website: 14 models, self-hosted artwork and fonts.');
